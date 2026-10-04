//! Streaming DPDFNet speech enhancer (Rust port of `dpdfnet.StreamEnhancer`).
//!
//! Per channel and per hop: causal STFT with a Vorbis window, one ONNX step
//! carrying the recurrent state, then ISTFT + overlap-add.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

/// The model's output lags its input by this many frames. Used to align the
/// noisy spectrum with the enhanced one (attenuation limit + visualisation).
const MODEL_DELAY_FRAMES: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum DpdfModelKind {
    #[default]
    DpdfNet2_48kHr,
    DpdfNet8_48kHr,
}

impl DpdfModelKind {
    pub fn label(self) -> &'static str {
        match self {
            DpdfModelKind::DpdfNet2_48kHr => "DPDFNet-2 48 kHz HR",
            DpdfModelKind::DpdfNet8_48kHr => "DPDFNet-8 48 kHz HR",
        }
    }

    /// Whether to run channels on separate threads. DPDFNet-8 takes ~5 ms per
    /// channel and hop, so left then right would not fit in the 10 ms hop.
    pub fn parallel(self) -> bool {
        match self {
            DpdfModelKind::DpdfNet2_48kHr => false,
            DpdfModelKind::DpdfNet8_48kHr => true,
        }
    }

    fn model_bytes(self) -> &'static [u8] {
        match self {
            DpdfModelKind::DpdfNet2_48kHr => {
                include_bytes!("../models/dpdfnet2_48khz_hr.onnx")
            }
            DpdfModelKind::DpdfNet8_48kHr => {
                include_bytes!("../models/dpdfnet8_48khz_hr.onnx")
            }
        }
    }
}

/// Model and STFT configuration read from the ONNX metadata.
struct ModelConfig {
    sr: usize,
    win_len: usize,
    hop_size: usize,
    n_freqs: usize,
    init_state: Vec<f32>,
}

/// Enhancer for one channel: owns its ONNX session so channels can run on
/// separate threads.
struct ChannelEnhancer {
    session: Session,
    in_spec_name: String,
    in_state_name: String,
    win_len: usize,
    hop_size: usize,
    n_freqs: usize,
    window: Arc<[f32]>,
    fft_fwd: Arc<dyn RealToComplex<f32>>,
    fft_inv: Arc<dyn ComplexToReal<f32>>,
    time_buf: Vec<f32>,
    freq_buf: Vec<Complex32>,
    spec_ri: Vec<f32>,
    scratch_fwd: Vec<Complex32>,
    scratch_inv: Vec<Complex32>,
    state: Vec<f32>,
    /// Last `win_len` input samples.
    in_buf: Vec<f32>,
    /// Overlap-add accumulator, `win_len` long.
    out_buf: Vec<f32>,
    /// Recent noisy spectra, so they can be aligned with the delayed model output.
    noisy_hist: VecDeque<Vec<Complex32>>,
    spec_noisy: Vec<Complex32>,
    spec_enh: Vec<Complex32>,
}

pub struct DpdfNet {
    channels: Vec<ChannelEnhancer>,
    init_state: Vec<f32>,
    /// Run channels on separate threads (for models too heavy to run them in turn).
    parallel: bool,
    pub sr: usize,
    pub win_len: usize,
    pub hop_size: usize,
    pub n_freqs: usize,
    pub ch: usize,
    /// Weight of the (aligned) noisy spectrum in the output, from the attenuation limit.
    atten_alpha: f32,
}

fn meta_usize(meta: &ort::session::ModelMetadata<'_>, key: &str) -> Result<usize> {
    meta.custom(key)
        .ok_or_else(|| anyhow!("DPDFNet model is missing metadata '{key}'"))?
        .trim()
        .parse()
        .with_context(|| format!("Invalid DPDFNet metadata '{key}'"))
}

fn meta_floats(meta: &ort::session::ModelMetadata<'_>, key: &str) -> Result<Vec<f32>> {
    meta.custom(key)
        .ok_or_else(|| anyhow!("DPDFNet model is missing metadata '{key}'"))?
        .split(',')
        .map(|v| v.trim().parse::<f32>())
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("Invalid DPDFNet metadata '{key}'"))
}

/// Single-threaded CPU session, like the Python package. Parallelism, where
/// needed, comes from running one session per channel on its own thread.
fn build_session(model_bytes: &[u8]) -> Result<Session> {
    let session = Session::builder()
        .and_then(|b| Ok(b.with_optimization_level(GraphOptimizationLevel::All)?))
        .and_then(|b| Ok(b.with_intra_threads(1)?))
        .and_then(|b| Ok(b.with_inter_threads(1)?))
        .and_then(|mut b| b.commit_from_memory(model_bytes))
        .map_err(|e| anyhow!("Failed to load DPDFNet model: {e}"))?;
    if session.inputs().len() < 2 || session.outputs().len() < 2 {
        return Err(anyhow!(
            "Expected streaming DPDFNet model with (spec, state) inputs and outputs"
        ));
    }
    Ok(session)
}

fn read_config(session: &Session) -> Result<ModelConfig> {
    let meta = session
        .metadata()
        .map_err(|e| anyhow!("Failed to read DPDFNet metadata: {e}"))?;
    let sr = meta_usize(&meta, "sample_rate")?;
    let win_len = meta_usize(&meta, "n_fft")?;
    let hop_size = meta_usize(&meta, "hop_length")?;
    let n_freqs = meta_usize(&meta, "freq_bins")?;
    let state_size = meta_usize(&meta, "state_size")?;
    let erb_norm_init = meta_floats(&meta, "erb_norm_init")?;
    let spec_norm_init = meta_floats(&meta, "spec_norm_init")?;
    if erb_norm_init.len() + spec_norm_init.len() > state_size {
        return Err(anyhow!("DPDFNet norm init does not fit in state"));
    }
    if n_freqs != win_len / 2 + 1 || hop_size * 2 != win_len {
        return Err(anyhow!(
            "Unsupported DPDFNet STFT config: n_fft={win_len}, hop={hop_size}, bins={n_freqs}"
        ));
    }
    let mut init_state = vec![0.0f32; state_size];
    init_state[..erb_norm_init.len()].copy_from_slice(&erb_norm_init);
    init_state[erb_norm_init.len()..erb_norm_init.len() + spec_norm_init.len()]
        .copy_from_slice(&spec_norm_init);
    Ok(ModelConfig {
        sr,
        win_len,
        hop_size,
        n_freqs,
        init_state,
    })
}

fn vorbis_window(win_len: usize) -> Vec<f32> {
    let half = win_len as f64 / 2.0;
    (0..win_len)
        .map(|i| {
            let s = (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / half).sin();
            (0.5 * std::f64::consts::PI * s * s).sin() as f32
        })
        .collect()
}

impl ChannelEnhancer {
    fn new(
        session: Session,
        cfg: &ModelConfig,
        window: Arc<[f32]>,
        fft_fwd: Arc<dyn RealToComplex<f32>>,
        fft_inv: Arc<dyn ComplexToReal<f32>>,
    ) -> Self {
        let (win_len, n_freqs) = (cfg.win_len, cfg.n_freqs);
        Self {
            in_spec_name: session.inputs()[0].name().to_string(),
            in_state_name: session.inputs()[1].name().to_string(),
            session,
            win_len,
            hop_size: cfg.hop_size,
            n_freqs,
            window,
            scratch_fwd: fft_fwd.make_scratch_vec(),
            scratch_inv: fft_inv.make_scratch_vec(),
            fft_fwd,
            fft_inv,
            time_buf: vec![0.0; win_len],
            freq_buf: vec![Complex32::default(); n_freqs],
            spec_ri: vec![0.0; n_freqs * 2],
            state: cfg.init_state.clone(),
            in_buf: vec![0.0; win_len],
            out_buf: vec![0.0; win_len],
            noisy_hist: VecDeque::with_capacity(MODEL_DELAY_FRAMES + 1),
            spec_noisy: vec![Complex32::default(); n_freqs],
            spec_enh: vec![Complex32::default(); n_freqs],
        }
    }

    fn reset(&mut self, init_state: &[f32]) {
        self.state.copy_from_slice(init_state);
        self.in_buf.fill(0.0);
        self.out_buf.fill(0.0);
        self.noisy_hist.clear();
        self.spec_noisy.fill(Complex32::default());
        self.spec_enh.fill(Complex32::default());
    }

    fn process(&mut self, input: &[f32], output: &mut [f32], atten_alpha: f32) -> Result<()> {
        let (win_len, hop, n_freqs) = (self.win_len, self.hop_size, self.n_freqs);

        // Analysis: slide the input window and run a causal STFT frame.
        self.in_buf.copy_within(hop.., 0);
        self.in_buf[win_len - hop..].copy_from_slice(input);
        for ((t, x), w) in self
            .time_buf
            .iter_mut()
            .zip(&self.in_buf)
            .zip(self.window.iter())
        {
            *t = x * w;
        }
        self.fft_fwd
            .process_with_scratch(
                &mut self.time_buf,
                &mut self.freq_buf,
                &mut self.scratch_fwd,
            )
            .map_err(|e| anyhow!("Forward FFT failed: {e}"))?;
        for (ri, x) in self.spec_ri.chunks_exact_mut(2).zip(&self.freq_buf) {
            ri[0] = x.re;
            ri[1] = x.im;
        }
        self.noisy_hist.push_back(self.freq_buf.clone());

        // Model step.
        let outputs = self
            .session
            .run(ort::inputs![
                self.in_spec_name.as_str() => TensorRef::from_array_view(([1usize, 1, n_freqs, 2], &self.spec_ri[..]))?,
                self.in_state_name.as_str() => TensorRef::from_array_view(([self.state.len()], &self.state[..]))?,
            ])
            .map_err(|e| anyhow!("DPDFNet inference failed: {e}"))?;
        let (_, spec_e) = outputs[0].try_extract_tensor::<f32>()?;
        let (_, state_out) = outputs[1].try_extract_tensor::<f32>()?;
        self.state.copy_from_slice(state_out);
        for (x, ri) in self.freq_buf.iter_mut().zip(spec_e.chunks_exact(2)) {
            *x = Complex32::new(ri[0], ri[1]);
        }
        drop(outputs);

        // Align the noisy spectrum with the delayed model output.
        if self.noisy_hist.len() > MODEL_DELAY_FRAMES {
            let noisy = self.noisy_hist.pop_front().unwrap();
            if atten_alpha > 0.0 {
                for (e, n) in self.freq_buf.iter_mut().zip(&noisy) {
                    *e = n * atten_alpha + *e * (1.0 - atten_alpha);
                }
            }
            self.spec_noisy = noisy;
        }
        self.spec_enh.copy_from_slice(&self.freq_buf);

        // Synthesis: ISTFT + overlap-add. The DC and Nyquist bins must be real.
        self.freq_buf[0].im = 0.0;
        self.freq_buf[n_freqs - 1].im = 0.0;
        self.fft_inv
            .process_with_scratch(
                &mut self.freq_buf,
                &mut self.time_buf,
                &mut self.scratch_inv,
            )
            .map_err(|e| anyhow!("Inverse FFT failed: {e}"))?;
        let norm = 1.0 / win_len as f32;
        for ((o, t), w) in self
            .out_buf
            .iter_mut()
            .zip(&self.time_buf)
            .zip(self.window.iter())
        {
            *o += t * norm * w;
        }
        // Vorbis window is power-complementary at 50% overlap, so the first hop is final.
        output.copy_from_slice(&self.out_buf[..hop]);
        self.out_buf.copy_within(hop.., 0);
        self.out_buf[win_len - hop..].fill(0.0);
        Ok(())
    }
}

impl DpdfNet {
    pub fn new(kind: DpdfModelKind, n_ch: usize) -> Result<Self> {
        let first = build_session(kind.model_bytes())?;
        let cfg = read_config(&first)?;

        let mut planner = RealFftPlanner::<f32>::new();
        let fft_fwd = planner.plan_fft_forward(cfg.win_len);
        let fft_inv = planner.plan_fft_inverse(cfg.win_len);
        let window: Arc<[f32]> = vorbis_window(cfg.win_len).into();

        let mut sessions = vec![first];
        for _ in 1..n_ch {
            sessions.push(build_session(kind.model_bytes())?);
        }
        let channels = sessions
            .into_iter()
            .map(|s| {
                ChannelEnhancer::new(s, &cfg, window.clone(), fft_fwd.clone(), fft_inv.clone())
            })
            .collect();

        log::info!(
            "Loaded {} (sr={}, n_fft={}, hop={}, {} channel(s){})",
            kind.label(),
            cfg.sr,
            cfg.win_len,
            cfg.hop_size,
            n_ch,
            if kind.parallel() { ", parallel" } else { "" }
        );

        Ok(Self {
            channels,
            parallel: kind.parallel() && n_ch > 1,
            sr: cfg.sr,
            win_len: cfg.win_len,
            hop_size: cfg.hop_size,
            n_freqs: cfg.n_freqs,
            ch: n_ch,
            init_state: cfg.init_state,
            atten_alpha: 0.0,
        })
    }

    /// Reset recurrent state and buffers of all channels.
    pub fn reset(&mut self) {
        for c in self.channels.iter_mut() {
            c.reset(&self.init_state);
        }
    }

    /// Attenuation limit in dB; 100 dB or more means no noisy signal is mixed back.
    pub fn set_atten_lim(&mut self, db: f32) {
        self.atten_alpha = if db >= 100.0 {
            0.0
        } else {
            10f32.powf(-db.max(0.0) / 20.0)
        };
    }

    /// Process one hop for all channels. `input` and `output` are planar,
    /// `ch * hop_size` samples long (channel `c` at `c * hop_size`).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let hop = self.hop_size;
        debug_assert_eq!(input.len(), self.ch * hop);
        debug_assert_eq!(output.len(), self.ch * hop);
        let alpha = self.atten_alpha;
        let jobs = self
            .channels
            .iter_mut()
            .zip(input.chunks_exact(hop))
            .zip(output.chunks_exact_mut(hop));
        if !self.parallel {
            for ((c, i), o) in jobs {
                c.process(i, o, alpha)?;
            }
            return Ok(());
        }
        // Channel 0 runs on this thread, the rest on scoped helper threads.
        std::thread::scope(|s| {
            let mut jobs = jobs;
            let ((c0, i0), o0) = jobs.next().expect("at least one channel");
            let helpers: Vec<_> = jobs
                .map(|((c, i), o)| s.spawn(move || c.process(i, o, alpha)))
                .collect();
            let mut result = c0.process(i0, o0, alpha);
            for h in helpers {
                let r = h
                    .join()
                    .unwrap_or_else(|_| Err(anyhow!("DPDFNet channel thread panicked")));
                result = result.and(r);
            }
            result
        })
    }

    /// Noisy spectrum of the last frame (aligned with `spec_enh`).
    pub fn spec_noisy(&self, c: usize) -> &[Complex32] {
        &self.channels[c].spec_noisy
    }

    /// Enhanced spectrum of the last frame.
    pub fn spec_enh(&self, c: usize) -> &[Complex32] {
        &self.channels[c].spec_enh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_db_attenuation_passes_input_through() {
        let mut m = DpdfNet::new(DpdfModelKind::DpdfNet2_48kHr, 2).unwrap();
        m.set_atten_lim(0.0);
        let hop = m.hop_size;
        let x: Vec<f32> = (0..hop * 2 * 40)
            .map(|i| 0.3 * (i as f32 * 0.031).sin() * (i as f32 * 0.0007).cos())
            .collect();
        let mut y = vec![0.0; x.len()];
        for (i, o) in x.chunks_exact(hop * 2).zip(y.chunks_exact_mut(hop * 2)) {
            m.process(i, o).unwrap();
        }
        // Output lags by one hop (window priming) plus the model delay. Check channel 0.
        let ch0 = |v: &[f32]| -> Vec<f32> {
            v.chunks_exact(hop * 2)
                .flat_map(|f| f[..hop].to_vec())
                .collect()
        };
        let (x0, y0) = (ch0(&x), ch0(&y));
        let delay = hop * (1 + MODEL_DELAY_FRAMES);
        let err = y0[delay..]
            .iter()
            .zip(&x0)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-4, "max error {err}");
    }
}
