//! Real-time capture pipeline for the Resemble Enhance denoiser.
//!
//! Kept separate from the DeepFilterNet and DPDFNet pipelines: it owns its own
//! device streams, worker thread and model. The model is mono and processes
//! `resemble::BLOCK`-sample blocks, so input is downmixed to mono and the
//! output is copied to every output channel.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Device, SampleRate, Stream, StreamConfig, SupportedStreamConfigRange};
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use ringbuf::HeapRb;
use rubato::{FftFixedIn, FftFixedOut, Resampler};

use crate::capture::{
    DeviceEvent, DeviceSelectError, DfControl, RbCons, RbProd, RecvControl, SendDeviceEvent,
    SendSpec, StreamStats,
};
use crate::resemble::{ResembleDenoiser, BLOCK, HOP, LOOKAHEAD, SAMPLE_RATE};

const SAMPLE_FORMAT: cpal::SampleFormat = cpal::SampleFormat::F32;
/// Spectrogram FFT size: two hops, so one display frame per model hop.
const SPEC_FFT: usize = 2 * HOP;
/// Smallest model time the output margin allows for, in seconds. With a run
/// every block the GPU stays clocked up: ~30 ms typical, ~50 ms worst on an
/// RTX 3060 laptop (idle GPUs between runs spike to ~180 ms).
const MIN_MARGIN_S: f32 = 0.06;
/// Input backlog tolerated before skipping ahead, in seconds.
const MAX_INPUT_BACKLOG_S: f32 = 0.5;

#[derive(Clone, Copy)]
enum Direction {
    Input,
    Output,
}

impl Direction {
    fn name(self) -> &'static str {
        match self {
            Direction::Input => "input",
            Direction::Output => "output",
        }
    }
}

fn find_device(device_str: Option<String>, dir: Direction) -> Result<Device, DeviceSelectError> {
    let Some(device_str) = device_str else {
        return Err(match dir {
            Direction::Input => DeviceSelectError::NoInputProvided,
            Direction::Output => DeviceSelectError::NoOutputProvided,
        });
    };
    let host = cpal::default_host();
    let devices = match dir {
        Direction::Input => host.input_devices(),
        Direction::Output => host.output_devices(),
    }
    .map_err(|e| DeviceSelectError::Other(e.into()))?;
    let needle = device_str.to_lowercase();
    for dev in devices {
        let name = dev.name().map_err(|e| DeviceSelectError::Other(e.into()))?;
        if name.to_lowercase().contains(&needle) {
            log::info!("Selected {} device: {}", dir.name(), name);
            return Ok(dev);
        }
    }
    log::error!("Requested {} device '{}' not found", dir.name(), device_str);
    Err(match dir {
        Direction::Input => DeviceSelectError::InputNotFound(device_str),
        Direction::Output => DeviceSelectError::OutputNotFound(device_str),
    })
}

/// Pick an f32 config, preferring the model rate.
fn stream_config(device: &Device, dir: Direction) -> Result<StreamConfig> {
    let all: Vec<SupportedStreamConfigRange> = match dir {
        Direction::Input => device.supported_input_configs()?.collect(),
        Direction::Output => device.supported_output_configs()?.collect(),
    };
    let configs: Vec<_> = all
        .into_iter()
        .filter(|c| c.sample_format() == SAMPLE_FORMAT)
        .collect();

    let sr = SampleRate(SAMPLE_RATE as u32);
    if let Some(c) = configs
        .iter()
        .find(|c| sr >= c.min_sample_rate() && sr <= c.max_sample_rate())
    {
        let mut c: StreamConfig = (*c).with_sample_rate(sr).into();
        c.buffer_size = BufferSize::Fixed(HOP as u32);
        log::info!("Selected {} config: {:?}", dir.name(), c);
        return Ok(c);
    }
    let c = configs
        .first()
        .ok_or_else(|| anyhow!("No suitable audio {} config found.", dir.name()))?;
    let mut c: StreamConfig = (*c).with_max_sample_rate().into();
    c.buffer_size = BufferSize::Fixed((HOP * c.sample_rate.0 as usize / SAMPLE_RATE) as u32);
    log::warn!("Using best matching {} config {:?}", dir.name(), c);
    Ok(c)
}

fn error_callback(
    dir: Direction,
    device_lost: Arc<AtomicBool>,
    device_event: Option<SendDeviceEvent>,
) -> impl FnMut(cpal::StreamError) {
    move |err| {
        log::error!("Error during audio {} {:?}", dir.name(), err);
        if let cpal::StreamError::DeviceNotAvailable = err {
            device_lost.store(true, Ordering::Relaxed);
            log::error!("Audio {} device lost, stopping stream.", dir.name());
            if let Some(ref sender) = device_event {
                let _ = sender.send(match dir {
                    Direction::Input => DeviceEvent::InputLost,
                    Direction::Output => DeviceEvent::OutputLost,
                });
            }
        }
    }
}

/// Input stream that downmixes to mono.
fn start_input(
    device: &Device,
    config: &StreamConfig,
    mut rb: RbProd,
    stats: Arc<StreamStats>,
    device_lost: Arc<AtomicBool>,
    device_event: Option<SendDeviceEvent>,
) -> Result<Stream> {
    let ch = config.channels as usize;
    let mut mono = Vec::new();
    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            mono.clear();
            mono.extend(
                data.chunks_exact(ch)
                    .map(|f| f.iter().sum::<f32>() / ch as f32),
            );
            if rb.push_slice(&mono) < mono.len() {
                stats.input_overflows.fetch_add(1, Ordering::Relaxed);
            }
            rb.sync();
        },
        error_callback(Direction::Input, device_lost, device_event),
        None,
    )?;
    stream.play()?;
    Ok(stream)
}

/// Output stream that plays the mono signal on every channel.
fn start_output(
    device: &Device,
    config: &StreamConfig,
    mut rb: RbCons,
    stats: Arc<StreamStats>,
    device_lost: Arc<AtomicBool>,
    device_event: Option<SendDeviceEvent>,
) -> Result<Stream> {
    let ch = config.channels as usize;
    let mut mono = Vec::new();
    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let frames = data.len() / ch;
            stats
                .out_callback_frames
                .fetch_max(frames, Ordering::Relaxed);
            mono.resize(frames, 0.0);
            let n = rb.pop_slice(&mut mono);
            mono[n..].fill(0.0);
            for (frame, &s) in data.chunks_exact_mut(ch).zip(&mono) {
                frame.fill(s);
            }
            if n < frames {
                stats.underruns.fetch_add(1, Ordering::Relaxed);
            }
        },
        error_callback(Direction::Output, device_lost, device_event),
        None,
    )?;
    stream.play()?;
    Ok(stream)
}

/// Spectrogram of a mono signal, one frame per hop, in the same dB scale as
/// the other pipelines.
struct SpecAnalyzer {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// Previous hop of samples, so frames span block boundaries.
    prev: Vec<f32>,
    time: Vec<f32>,
    freq: Vec<Complex32>,
    scratch: Vec<Complex32>,
}

impl SpecAnalyzer {
    fn new(planner: &mut RealFftPlanner<f32>) -> Self {
        let fft = planner.plan_fft_forward(SPEC_FFT);
        let window = (0..SPEC_FFT)
            .map(|i| {
                let s = (std::f32::consts::PI * i as f32 / SPEC_FFT as f32).sin();
                s * s
            })
            .collect();
        Self {
            time: fft.make_input_vec(),
            freq: fft.make_output_vec(),
            scratch: fft.make_scratch_vec(),
            fft,
            window,
            prev: vec![0.0; HOP],
        }
    }

    fn push(&mut self, block: &[f32], sender: &SendSpec) {
        // Same scaling as libDF's STFT (`1 / (n_fft^2 / (2 * hop))`).
        let wnorm = 2.0 * HOP as f32 / (SPEC_FFT * SPEC_FFT) as f32;
        for hop in block.chunks_exact(HOP) {
            self.time[..HOP].copy_from_slice(&self.prev);
            self.time[HOP..].copy_from_slice(hop);
            self.prev.copy_from_slice(hop);
            for (t, w) in self.time.iter_mut().zip(&self.window) {
                *t *= w;
            }
            if self
                .fft
                .process_with_scratch(&mut self.time, &mut self.freq, &mut self.scratch)
                .is_err()
            {
                return;
            }
            let spec: Box<[f32]> = self
                .freq
                .iter()
                .map(|x| ((x * wnorm).norm_sqr()).max(1e-10).log10() * 10.0)
                .collect();
            let _ = sender.send(spec);
        }
    }
}

/// Transient smoother, same tuning as the other pipelines.
struct Smoother {
    env_attack: f32,
    env_release: f32,
    gain_attack: f32,
    gain_release: f32,
    env: f32,
    gain: f32,
}

impl Smoother {
    const SPIKE_THRESH_DB: f32 = 6.0;

    fn new(sr: usize) -> Self {
        let coeff = |tau_s: f32| (-1.0 / (tau_s * sr as f32)).exp();
        Self {
            env_attack: coeff(0.030),
            env_release: coeff(0.300),
            gain_attack: coeff(0.003),
            gain_release: coeff(0.080),
            env: -1.0,
            gain: 1.0,
        }
    }

    fn process(&mut self, samples: &mut [f32], enabled: bool) {
        for s in samples.iter_mut() {
            let level = s.abs();
            if self.env < 0.0 {
                self.env = level.max(1e-8);
            }
            let c = if level > self.env {
                self.env_attack
            } else {
                self.env_release
            };
            self.env = self.env * c + level * (1.0 - c);
            let level_db = 20.0 * (level.max(1e-8) / self.env.max(1e-8)).log10();
            let target = if level_db > Self::SPIKE_THRESH_DB {
                10f32.powf(-(level_db - Self::SPIKE_THRESH_DB) / 20.0)
            } else {
                1.0
            };
            let c = if target < self.gain {
                self.gain_attack
            } else {
                self.gain_release
            };
            self.gain = self.gain * c + target * (1.0 - c);
            if enabled {
                *s *= self.gain;
            }
        }
    }
}

struct Worker {
    model: ResembleDenoiser,
    rb_in: RbCons,
    rb_out: RbProd,
    input_sr: usize,
    output_sr: usize,
    should_stop: Arc<AtomicBool>,
    device_lost: Arc<AtomicBool>,
    s_spec: Option<(SendSpec, SendSpec)>,
    r_opt: Option<RecvControl>,
    shared_volume: Arc<AtomicU32>,
    smoother_enabled: Arc<AtomicBool>,
    delay_ms: Arc<AtomicU32>,
    stats: Arc<StreamStats>,
}

impl Worker {
    fn run(mut self) {
        let mut input_resampler = (self.input_sr != SAMPLE_RATE).then(|| {
            FftFixedOut::<f32>::new(self.input_sr, SAMPLE_RATE, HOP, 1, 1)
                .expect("Failed to init input resampler")
        });
        let mut output_resampler = (self.output_sr != SAMPLE_RATE).then(|| {
            FftFixedIn::<f32>::new(SAMPLE_RATE, self.output_sr, HOP, 1, 1)
                .expect("Failed to init output resampler")
        });
        let mut resample_in = vec![
            0.0f32;
            input_resampler
                .as_ref()
                .map_or(HOP, |r| r.input_frames_max())
        ];
        let mut resample_out = vec![vec![0.0f32; HOP]];
        let mut resampled_block = vec![vec![
            0.0f32;
            output_resampler
                .as_ref()
                .map_or(HOP, |r| r.output_frames_max())
        ]];
        let mut block_in: Vec<f32> = Vec::with_capacity(BLOCK);
        let mut block_out = vec![0.0f32; BLOCK];
        let mut out = Vec::with_capacity(BLOCK * 2);

        let mut planner = RealFftPlanner::<f32>::new();
        let mut spec_noisy = SpecAnalyzer::new(&mut planner);
        let mut spec_enh = SpecAnalyzer::new(&mut planner);
        let mut smoother = Smoother::new(self.output_sr);
        // Recent worst model time, to size the output margin. DirectML times
        // jitter with GPU clocks.
        let mut compute_peak = 0.0f32;
        let mut dropped = 0usize;
        let mut last_report = Instant::now();

        log::info!(
            "Resemble Enhance worker started (input {} Hz, output {} Hz, model {} Hz)",
            self.input_sr,
            self.output_sr,
            SAMPLE_RATE
        );
        while !self.should_stop.load(Ordering::Relaxed) && !self.device_lost.load(Ordering::Relaxed)
        {
            if let Some(r_opt) = self.r_opt.as_ref() {
                while let Ok((c, v)) = r_opt.try_recv() {
                    if c == DfControl::AttenLim {
                        self.model.set_atten_lim(v);
                    }
                }
            }

            // Gather one hop at the model rate.
            let n_in = input_resampler
                .as_ref()
                .map_or(HOP, |r| r.input_frames_next());
            let queued = self.rb_in.len();
            if queued < n_in {
                sleep(Duration::from_millis(1));
                continue;
            }
            // Fell behind (e.g. a stall): skip old input instead of staying late.
            let max_queued = n_in + (MAX_INPUT_BACKLOG_S * self.input_sr as f32) as usize;
            if queued > max_queued {
                self.rb_in.skip(queued - n_in);
                dropped += 1;
            }
            let n = self.rb_in.pop_slice(&mut resample_in[..n_in]);
            debug_assert_eq!(n, n_in);
            match input_resampler.as_mut() {
                Some(r) => {
                    r.process_into_buffer(&[&resample_in[..n_in]], &mut resample_out, None)
                        .expect("Input resampling failed");
                    block_in.extend_from_slice(&resample_out[0][..HOP]);
                }
                None => block_in.extend_from_slice(&resample_in[..HOP]),
            }
            if block_in.len() < BLOCK {
                continue;
            }

            let t0 = Instant::now();
            if let Err(e) = self.model.process(&block_in, &mut block_out) {
                log::error!("Resemble Enhance processing failed: {e:#}");
                block_out.fill(0.0);
            }
            let elapsed = t0.elapsed().as_secs_f32();
            // Decays with a half-life of about 30 s.
            let decay = 0.5f32.powf(BLOCK as f32 / SAMPLE_RATE as f32 / 30.0);
            compute_peak = elapsed.max(compute_peak * decay);
            block_in.clear();

            if let Some((ref s_noisy, ref s_enh)) = self.s_spec {
                spec_noisy.push(self.model.input_block(), s_noisy);
                spec_enh.push(&block_out, s_enh);
            }

            out.clear();
            match output_resampler.as_mut() {
                Some(r) => {
                    for hop in block_out.chunks_exact(HOP) {
                        let (_, n) = r
                            .process_into_buffer(&[hop], &mut resampled_block, None)
                            .expect("Output resampling failed");
                        out.extend_from_slice(&resampled_block[0][..n]);
                    }
                }
                None => out.extend_from_slice(&block_out),
            }
            smoother.process(&mut out, self.smoother_enabled.load(Ordering::Relaxed));
            let vol = f32::from_bits(self.shared_volume.load(Ordering::Relaxed));
            for s in out.iter_mut() {
                *s *= vol;
            }

            // Output arrives in blocks, so keep a margin queued for the time the
            // model takes. Refill after running dry; trim when well above the margin.
            let cb = self
                .stats
                .out_callback_frames
                .load(Ordering::Relaxed)
                .max(HOP);
            let margin_s = compute_peak.max(MIN_MARGIN_S) + 0.02;
            let target = 2 * cb + (margin_s * self.output_sr as f32) as usize;
            // Postponed producer: sync to see what the device consumed since the last push.
            self.rb_out.sync();
            let queued = self.rb_out.len();
            let mut start = 0;
            // Audio queued ahead of this block once it is pushed.
            let mut ahead = queued;
            if queued < cb {
                let silence = vec![0.0; target - queued];
                self.rb_out.push_slice(&silence);
                ahead = target;
            } else if queued > target + out.len() / 4 {
                start = (queued - target).min(out.len());
                ahead = target;
                dropped += 1;
            }
            // A sample waits for its block and the lookahead, then the model,
            // then the audio queued ahead of it.
            let delay_s = (BLOCK + LOOKAHEAD) as f32 / SAMPLE_RATE as f32
                + elapsed
                + ahead as f32 / self.output_sr as f32;
            self.delay_ms
                .store((delay_s * 1000.0) as u32, Ordering::Relaxed);
            self.rb_out.push_slice(&out[start..]);
            self.rb_out.sync();

            if last_report.elapsed() >= Duration::from_secs(5)
                && log::log_enabled!(log::Level::Debug)
            {
                last_report = Instant::now();
                log::debug!(
                    "Resemble: model {:.0} ms/block (peak {:.0} ms), output queue {:.0} ms \
                     (callback {} frames; underruns {}, input overflows {}, drops {})",
                    elapsed * 1000.0,
                    compute_peak * 1000.0,
                    self.rb_out.len() as f32 * 1000.0 / self.output_sr as f32,
                    cb,
                    self.stats.underruns.load(Ordering::Relaxed),
                    self.stats.input_overflows.load(Ordering::Relaxed),
                    dropped
                );
            }
        }
        log::info!("Resemble Enhance worker stopped");
    }
}

pub struct ResembleCapture {
    pub sr: usize,
    pub frame_size: usize,
    pub freq_size: usize,
    /// Whether the denoiser runs on the GPU (DirectML) rather than the CPU.
    pub on_gpu: bool,
    /// Current estimate of the input-to-output delay.
    pub delay_ms: Arc<AtomicU32>,
    should_stop: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
    input_stream: Option<Stream>,
    output_stream: Option<Stream>,
}

impl ResembleCapture {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_device: Option<String>,
        output_device: Option<String>,
        s_noisy: Option<SendSpec>,
        s_enh: Option<SendSpec>,
        r_opt: Option<RecvControl>,
        s_device_event: Option<SendDeviceEvent>,
        shared_volume: Arc<AtomicU32>,
        smoother_enabled: Arc<AtomicBool>,
    ) -> Result<Self> {
        let in_dev = find_device(input_device, Direction::Input)?;
        let out_dev = find_device(output_device, Direction::Output)?;
        let in_cfg = stream_config(&in_dev, Direction::Input)?;
        let out_cfg = stream_config(&out_dev, Direction::Output)?;

        let mut model = ResembleDenoiser::new()?;
        let on_gpu = model.on_gpu();
        let delay_ms = Arc::new(AtomicU32::new(0));
        // Warm up: the first runs pay session initialisation (and are slow on DirectML).
        let mut warm = vec![0.0f32; BLOCK];
        for _ in 0..2 {
            model.process(&vec![0.0; BLOCK], &mut warm)?;
        }
        model.reset();

        let in_sr = in_cfg.sample_rate.0 as usize;
        let out_sr = out_cfg.sample_rate.0 as usize;
        // Two seconds each way, well above the backlog and margin limits.
        let (in_prod, in_cons) = HeapRb::<f32>::new(in_sr * 2).split();
        let (out_prod, out_cons) = HeapRb::<f32>::new(out_sr * 2).split();

        let should_stop = Arc::new(AtomicBool::new(false));
        let device_lost = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(StreamStats::default());
        let worker = Worker {
            model,
            rb_in: in_cons,
            rb_out: out_prod.into_postponed(),
            input_sr: in_sr,
            output_sr: out_sr,
            should_stop: should_stop.clone(),
            device_lost: device_lost.clone(),
            s_spec: s_noisy.zip(s_enh),
            r_opt,
            shared_volume,
            smoother_enabled,
            delay_ms: delay_ms.clone(),
            stats: stats.clone(),
        };
        let worker_handle = Some(
            thread::Builder::new()
                .name("resemble-worker".into())
                .spawn(move || worker.run())?,
        );

        // Output first: if input started first, audio would queue up while the
        // output device opens.
        let output_stream = start_output(
            &out_dev,
            &out_cfg,
            out_cons,
            stats.clone(),
            device_lost.clone(),
            s_device_event.clone(),
        )?;
        let input_stream = start_input(
            &in_dev,
            &in_cfg,
            in_prod.into_postponed(),
            stats,
            device_lost,
            s_device_event,
        )?;
        log::info!("Resemble Enhance capture started");

        Ok(Self {
            sr: SAMPLE_RATE,
            frame_size: HOP,
            freq_size: SPEC_FFT / 2 + 1,
            on_gpu,
            delay_ms,
            should_stop,
            worker_handle,
            input_stream: Some(input_stream),
            output_stream: Some(output_stream),
        })
    }

    pub fn should_stop(&mut self) -> Result<()> {
        // Pausing may fail if the device is already gone; that's fine.
        if let Some(s) = self.output_stream.as_ref() {
            let _ = s.pause();
        }
        if let Some(s) = self.input_stream.as_ref() {
            let _ = s.pause();
        }
        if let Some(h) = self.worker_handle.take() {
            log::info!("Joining Resemble Enhance worker");
            self.should_stop.store(true, Ordering::Relaxed);
            let _ = h.join();
        }
        Ok(())
    }
}

impl Drop for ResembleCapture {
    fn drop(&mut self) {
        let _ = self.should_stop();
    }
}
