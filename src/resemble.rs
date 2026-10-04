//! Streaming Resemble Enhance denoiser.
//!
//! The ONNX graph comes from `scripts/export_resemble_enhance.py`. It is not
//! causal and takes a fixed `CHUNK`-sample window, so the stream is processed in
//! overlapping windows: every block of new samples the window slides and the
//! model runs again. The block just before the last `LOOKAHEAD` samples (future
//! context) is emitted, crossfaded with the previous run.
//!
//! Small blocks cost compute but no quality (the window stays the same);
//! shorter lookahead does cost quality. Measured on speech mixed with music
//! (SI-SDR vs. clean speech): 457 ms block / 152 ms lookahead 11.0 dB,
//! 95 / 152 ms 10.8 dB, 95 / 76 ms 10.1 dB, 95 / 38 ms 9.1 dB.

use anyhow::{anyhow, Context, Result};
use ort::ep;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;

pub const SAMPLE_RATE: usize = 44_100;
/// STFT hop of the model.
pub const HOP: usize = 420;
/// Frames per model window; must match the exported graph.
const CHUNK_FRAMES: usize = 128;
/// New frames per model run. On the GPU a run takes ~30 ms, so short blocks
/// keep the delay low (and the GPU clocked up). On the CPU a run takes ~220 ms,
/// so blocks must be long enough to keep up.
const GPU_BLOCK_FRAMES: usize = 10;
const CPU_BLOCK_FRAMES: usize = 48;
/// Future context frames after the emitted block.
const LOOKAHEAD_FRAMES: usize = 16;

const CHUNK: usize = CHUNK_FRAMES * HOP;
pub const LOOKAHEAD: usize = LOOKAHEAD_FRAMES * HOP;

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
    /// New samples per run.
    block: usize,
    /// Crossfade with the previous run: no longer than the emitted block.
    xfade: usize,
    /// Offset of the emitted block within the window.
    emit_start: usize,
    /// Last `CHUNK` input samples.
    window: Vec<f32>,
    /// Output of the previous run for the next crossfade.
    prev_tail: Vec<f32>,
    /// Weight of the (aligned) input in the output, from the attenuation limit.
    atten_alpha: f32,
}

impl ResembleDenoiser {
    /// Loads the denoiser on the GPU when `use_gpu` is set and DirectML is
    /// available, otherwise on the CPU.
    pub fn new(use_gpu: bool) -> Result<Self> {
        let gpu_session = if use_gpu {
            build_session(true)
                .map_err(|e| log::warn!("DirectML unavailable for Resemble Enhance, using CPU: {e:#}"))
                .ok()
        } else {
            None
        };
        let on_gpu = gpu_session.is_some();
        let session = match gpu_session {
            Some(s) => s,
            None => build_session(false)?,
        };
        let block = if on_gpu { GPU_BLOCK_FRAMES } else { CPU_BLOCK_FRAMES } * HOP;
        let xfade = LOOKAHEAD.min(block);
        log::info!(
            "Loaded Resemble Enhance denoiser on {} (window {} ms, block {} ms, lookahead {} ms)",
            if on_gpu { "GPU (DirectML)" } else { "CPU" },
            CHUNK * 1000 / SAMPLE_RATE,
            block * 1000 / SAMPLE_RATE,
            LOOKAHEAD * 1000 / SAMPLE_RATE
        );
        Ok(Self {
            session,
            on_gpu,
            block,
            xfade,
            emit_start: CHUNK - LOOKAHEAD - block,
            window: vec![0.0; CHUNK],
            prev_tail: vec![0.0; xfade],
            atten_alpha: 0.0,
        })
    }

    pub fn on_gpu(&self) -> bool {
        self.on_gpu
    }

    /// New samples per `process` call.
    pub fn block(&self) -> usize {
        self.block
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
        &self.window[self.emit_start..self.emit_start + self.block]
    }

    /// Takes `block()` new mono samples and writes as many processed samples,
    /// `LOOKAHEAD` samples behind the input (plus the block itself in wall-clock
    /// latency).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let (block, xfade, emit) = (self.block, self.xfade, self.emit_start);
        debug_assert_eq!(input.len(), block);
        debug_assert_eq!(output.len(), block);
        self.window.copy_within(block.., 0);
        self.window[CHUNK - block..].copy_from_slice(input);

        let outputs = self
            .session
            .run(ort::inputs![
                "wav" => TensorRef::from_array_view(([1usize, CHUNK], &self.window[..]))?,
            ])
            .map_err(|e| anyhow!("Resemble Enhance denoiser failed: {e}"))?;
        let (_, y) = outputs[0].try_extract_tensor::<f32>()?;

        output.copy_from_slice(&y[emit..emit + block]);
        // Linear crossfade with the previous run, as upstream's chunk merging does.
        for (i, (o, p)) in output.iter_mut().zip(&self.prev_tail).enumerate() {
            let w = (i as f32 + 0.5) / xfade as f32;
            *o = p * (1.0 - w) + *o * w;
        }
        // The next block starts where this run's lookahead does.
        self.prev_tail
            .copy_from_slice(&y[CHUNK - LOOKAHEAD..CHUNK - LOOKAHEAD + xfade]);
        drop(outputs);

        let alpha = self.atten_alpha;
        if alpha > 0.0 {
            for (o, x) in output.iter_mut().zip(&self.window[emit..]) {
                *o = x * alpha + *o * (1.0 - alpha);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(m: &mut ResembleDenoiser, x: &[f32]) -> Vec<f32> {
        let b = m.block();
        let mut y = vec![0.0; x.len() / b * b];
        for (i, o) in x.chunks_exact(b).zip(y.chunks_exact_mut(b)) {
            m.process(i, o).unwrap();
        }
        y
    }

    #[test]
    fn zero_db_attenuation_passes_input_through() {
        for use_gpu in [true, false] {
            let mut m = ResembleDenoiser::new(use_gpu).unwrap();
            m.set_atten_lim(0.0);
            let x: Vec<f32> = (0..m.block() * 6)
                .map(|i| 0.3 * (i as f32 * 0.031).sin() * (i as f32 * 0.0007).cos())
                .collect();
            let y = run(&mut m, &x);
            let err = y[LOOKAHEAD..]
                .iter()
                .zip(&x)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(err < 1e-6, "max error {err} (gpu {use_gpu})");
        }
    }

    /// Digital silence once produced NaN on the CPU (0 / 0 in the phase).
    #[test]
    fn silence_then_signal_stays_finite_on_cpu_and_gpu() {
        for use_gpu in [false, true] {
            let mut m = ResembleDenoiser::new(use_gpu).unwrap();
            let x: Vec<f32> = (0..SAMPLE_RATE * 3)
                .map(|i| {
                    if i < SAMPLE_RATE {
                        0.0
                    } else {
                        0.2 * (i as f32 * 0.05).sin() * (i as f32 * 0.0003).sin()
                    }
                })
                .collect();
            let y = run(&mut m, &x);
            assert!(
                y.iter().all(|v| v.is_finite() && v.abs() <= 1.0),
                "bad output (gpu {})",
                m.on_gpu()
            );
        }
    }
}
