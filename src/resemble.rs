//! Streaming Resemble Enhance denoiser.
//!
//! The ONNX graph comes from `scripts/export_resemble_enhance.py`. It is not
//! causal and takes a fixed `CHUNK`-sample window, so the stream is processed in
//! overlapping windows: every `BLOCK` new samples the window slides and the
//! model runs again. The block just before the last `LOOKAHEAD` samples (future
//! context) is emitted, crossfaded with the previous run over `LOOKAHEAD` samples.

use anyhow::{anyhow, Context, Result};
use ort::ep;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

pub const SAMPLE_RATE: usize = 44_100;
/// STFT hop of the model.
pub const HOP: usize = 420;
/// Frames per model window; must match the exported graph.
const CHUNK_FRAMES: usize = 128;
/// New frames per model run.
const BLOCK_FRAMES: usize = 48;
/// Future context frames after the emitted block; also the crossfade length.
const LOOKAHEAD_FRAMES: usize = 16;

const CHUNK: usize = CHUNK_FRAMES * HOP;
pub const BLOCK: usize = BLOCK_FRAMES * HOP;
pub const LOOKAHEAD: usize = LOOKAHEAD_FRAMES * HOP;
/// Offset of the emitted block within the window.
const EMIT_START: usize = CHUNK - LOOKAHEAD - BLOCK;

const MODEL_BYTES: &[u8] = include_bytes!("../models/resemble_denoiser.onnx");

/// Builds a session on the GPU via DirectML, or on the CPU.
///
/// DirectML graph fusion is disabled: it fails to initialise on this graph,
/// and with it off DirectML runs it fine.
fn build_session(gpu: bool) -> Result<Session> {
    let mut builder = ort_ok(Session::builder())?;
    if gpu {
        builder = ort_ok(
            builder.with_execution_providers([ep::DirectML::default().build().error_on_failure()]),
        )?;
        builder = ort_ok(builder.with_memory_pattern(false))?;
        builder = ort_ok(builder.with_parallel_execution(false))?;
        builder = ort_ok(builder.with_config_entry("ep.dml.disable_graph_fusion", "1"))?;
    } else {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(4))
            .unwrap_or(1);
        builder = ort_ok(builder.with_intra_threads(threads))?;
    }
    builder = ort_ok(builder.with_optimization_level(GraphOptimizationLevel::All))?;
    let session = builder
        .commit_from_memory(MODEL_BYTES)
        .map_err(|e| anyhow!("Failed to load the Resemble Enhance denoiser: {e}"))?;
    check_metadata(&session).context("Unexpected Resemble Enhance denoiser model")?;
    Ok(session)
}

fn ort_ok<T, E: std::fmt::Display>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| anyhow!("{e}"))
}

fn check_metadata(session: &Session) -> Result<()> {
    let meta = session.metadata().map_err(|e| anyhow!("{e}"))?;
    for (key, want) in [
        ("sample_rate", SAMPLE_RATE),
        ("hop_length", HOP),
        ("chunk_frames", CHUNK_FRAMES),
    ] {
        let got: usize = meta
            .custom(key)
            .ok_or_else(|| anyhow!("missing metadata '{key}'"))?
            .trim()
            .parse()
            .with_context(|| format!("invalid metadata '{key}'"))?;
        if got != want {
            return Err(anyhow!("metadata '{key}' is {got}, expected {want}"));
        }
    }
    Ok(())
}

pub struct ResembleDenoiser {
    session: Session,
    on_gpu: bool,
    /// Last `CHUNK` input samples.
    window: Vec<f32>,
    /// Output of the previous run for the next crossfade.
    prev_tail: Vec<f32>,
    /// Weight of the (aligned) input in the output, from the attenuation limit.
    atten_alpha: f32,
}

impl ResembleDenoiser {
    /// Loads the denoiser, on the GPU if DirectML is available.
    pub fn new() -> Result<Self> {
        let (session, on_gpu) = match build_session(true) {
            Ok(s) => (s, true),
            Err(e) => {
                log::warn!("DirectML unavailable for Resemble Enhance, using CPU: {e:#}");
                (build_session(false)?, false)
            }
        };
        log::info!(
            "Loaded Resemble Enhance denoiser on {} (window {} ms, block {} ms, lookahead {} ms)",
            if on_gpu { "GPU (DirectML)" } else { "CPU" },
            CHUNK * 1000 / SAMPLE_RATE,
            BLOCK * 1000 / SAMPLE_RATE,
            LOOKAHEAD * 1000 / SAMPLE_RATE
        );
        Ok(Self {
            session,
            on_gpu,
            window: vec![0.0; CHUNK],
            prev_tail: vec![0.0; LOOKAHEAD],
            atten_alpha: 0.0,
        })
    }

    pub fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    /// Clears the audio history.
    pub fn reset(&mut self) {
        self.window.fill(0.0);
        self.prev_tail.fill(0.0);
    }

    /// Attenuation limit in dB; 100 dB or more means no input is mixed back.
    pub fn set_atten_lim(&mut self, db: f32) {
        self.atten_alpha = if db >= 100.0 {
            0.0
        } else {
            10f32.powf(-db.max(0.0) / 20.0)
        };
    }

    /// Input samples aligned with the last block written by `process`.
    pub fn input_block(&self) -> &[f32] {
        &self.window[EMIT_START..EMIT_START + BLOCK]
    }

    /// Takes `BLOCK` new mono samples and writes `BLOCK` processed samples,
    /// `LOOKAHEAD` samples behind the input (plus the block itself in wall-clock
    /// latency).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        debug_assert_eq!(input.len(), BLOCK);
        debug_assert_eq!(output.len(), BLOCK);
        self.window.copy_within(BLOCK.., 0);
        self.window[CHUNK - BLOCK..].copy_from_slice(input);

        let outputs = self
            .session
            .run(ort::inputs![
                "wav" => TensorRef::from_array_view(([1usize, CHUNK], &self.window[..]))?,
            ])
            .map_err(|e| anyhow!("Resemble Enhance denoiser failed: {e}"))?;
        let (_, y) = outputs[0].try_extract_tensor::<f32>()?;

        output.copy_from_slice(&y[EMIT_START..EMIT_START + BLOCK]);
        // Linear crossfade with the previous run, as upstream's chunk merging does.
        for (i, (o, p)) in output.iter_mut().zip(&self.prev_tail).enumerate() {
            let w = (i as f32 + 0.5) / LOOKAHEAD as f32;
            *o = p * (1.0 - w) + *o * w;
        }
        self.prev_tail.copy_from_slice(&y[CHUNK - LOOKAHEAD..]);
        drop(outputs);

        let alpha = self.atten_alpha;
        if alpha > 0.0 {
            for (o, x) in output.iter_mut().zip(&self.window[EMIT_START..]) {
                *o = x * alpha + *o * (1.0 - alpha);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_db_attenuation_passes_input_through() {
        let mut m = ResembleDenoiser::new().unwrap();
        m.set_atten_lim(0.0);
        let x: Vec<f32> = (0..BLOCK * 6)
            .map(|i| 0.3 * (i as f32 * 0.031).sin() * (i as f32 * 0.0007).cos())
            .collect();
        let mut y = vec![0.0; x.len()];
        for (i, o) in x.chunks_exact(BLOCK).zip(y.chunks_exact_mut(BLOCK)) {
            m.process(i, o).unwrap();
        }
        let err = y[LOOKAHEAD..]
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-6, "max error {err}");
    }

    #[test]
    fn output_is_finite_and_bounded() {
        let mut m = ResembleDenoiser::new().unwrap();
        let x: Vec<f32> = (0..BLOCK * 4)
            .map(|i| 0.2 * (i as f32 * 0.05).sin())
            .collect();
        let mut y = vec![0.0; x.len()];
        for (i, o) in x.chunks_exact(BLOCK).zip(y.chunks_exact_mut(BLOCK)) {
            m.process(i, o).unwrap();
        }
        assert!(y.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    }
}
