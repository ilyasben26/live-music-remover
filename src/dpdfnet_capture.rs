//! Real-time capture pipeline for DPDFNet models.
//!
//! Kept separate from the DeepFilterNet pipeline in `capture.rs`: it owns its
//! own device streams, worker thread and model instance.

use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Device, SampleRate, Stream, StreamConfig, SupportedStreamConfigRange};
use realfft::num_complex::Complex32;
use ringbuf::HeapRb;
use rubato::{FftFixedIn, FftFixedOut, Resampler};

use crate::capture::{
    max_output_queue, DeviceEvent, DeviceSelectError, DfControl, RbCons, RbProd, RecvControl,
    SendDeviceEvent, SendSpec, StreamStats,
};
use crate::dpdfnet::{DpdfModelKind, DpdfNet};
use crate::noise::{NoiseControls, NoiseInjector};

const SAMPLE_FORMAT: cpal::SampleFormat = cpal::SampleFormat::F32;
const PROCESS_CHANNELS: usize = 2;

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

/// Pick a stereo (then mono, then multi-channel) f32 config, preferring the model rate.
fn stream_config(
    device: &Device,
    sample_rate: u32,
    frame_size: usize,
    dir: Direction,
) -> Result<StreamConfig> {
    let all: Vec<SupportedStreamConfigRange> = match dir {
        Direction::Input => device.supported_input_configs()?.collect(),
        Direction::Output => device.supported_output_configs()?.collect(),
    };
    let f32_cfgs = || all.iter().filter(|c| c.sample_format() == SAMPLE_FORMAT);
    let mut configs: Vec<_> = f32_cfgs()
        .filter(|c| c.channels() == PROCESS_CHANNELS as u16)
        .collect();
    configs.extend(f32_cfgs().filter(|c| c.channels() == 1));
    configs.extend(f32_cfgs().filter(|c| c.channels() > PROCESS_CHANNELS as u16));

    let sr = SampleRate(sample_rate);
    if let Some(c) = configs
        .iter()
        .find(|c| sr >= c.min_sample_rate() && sr <= c.max_sample_rate())
    {
        let mut c: StreamConfig = (*c).with_sample_rate(sr).into();
        c.buffer_size = BufferSize::Fixed(frame_size as u32);
        log::info!("Selected {} config: {:?}", dir.name(), c);
        return Ok(c);
    }
    let c = configs
        .first()
        .ok_or_else(|| anyhow!("No suitable audio {} config found.", dir.name()))?;
    let mut c: StreamConfig = (*c).with_max_sample_rate().into();
    c.buffer_size = BufferSize::Fixed(frame_size as u32 * c.sample_rate.0 / sample_rate);
    log::warn!("Using best matching {} config {:?}", dir.name(), c);
    Ok(c)
}

fn remap_interleaved_channels(
    input: &[f32],
    input_ch: usize,
    output: &mut [f32],
    output_ch: usize,
) {
    for (in_frame, out_frame) in input
        .chunks_exact(input_ch)
        .zip(output.chunks_exact_mut(output_ch))
    {
        for (ch, o) in out_frame.iter_mut().enumerate() {
            *o = if ch < input_ch {
                in_frame[ch]
            } else {
                in_frame[0]
            };
        }
    }
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

fn start_input(
    device: &Device,
    config: &StreamConfig,
    mut rb: RbProd,
    stats: Arc<StreamStats>,
    device_lost: Arc<AtomicBool>,
    device_event: Option<SendDeviceEvent>,
) -> Result<Stream> {
    let input_ch = config.channels as usize;
    let model_ch = PROCESS_CHANNELS;
    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mapped;
            let data = if input_ch == model_ch {
                data
            } else {
                let mut m = vec![0.0; data.len() / input_ch * model_ch];
                remap_interleaved_channels(data, input_ch, &mut m, model_ch);
                mapped = m;
                &mapped[..]
            };
            if rb.push_slice(data) < data.len() {
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

fn start_output(
    device: &Device,
    config: &StreamConfig,
    mut rb: RbCons,
    stats: Arc<StreamStats>,
    device_lost: Arc<AtomicBool>,
    device_event: Option<SendDeviceEvent>,
) -> Result<Stream> {
    let output_ch = config.channels as usize;
    let model_ch = PROCESS_CHANNELS;
    let stream = device.build_output_stream(
        config,
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            stats
                .out_callback_frames
                .fetch_max(data.len() / output_ch, Ordering::Relaxed);
            let complete = if output_ch == model_ch {
                let n = rb.pop_slice(data);
                data[n..].fill(0.0);
                n == data.len()
            } else {
                let mut rb_data = vec![0.0; data.len() / output_ch * model_ch];
                let n = rb.pop_slice(&mut rb_data);
                remap_interleaved_channels(&rb_data, model_ch, data, output_ch);
                n == rb_data.len()
            };
            if !complete {
                stats.underruns.fetch_add(1, Ordering::Relaxed);
            }
        },
        error_callback(Direction::Output, device_lost, device_event),
        None,
    )?;
    stream.play()?;
    Ok(stream)
}

/// Same scaling as libDF's STFT (`1 / (n_fft^2 / (2 * hop))`), so spectrograms
/// share the DeepFilterNet display range.
fn push_spec(specs: &[&[Complex32]], wnorm: f32, sender: &SendSpec) {
    let n_ch = specs.len() as f32;
    let n_freqs = specs[0].len();
    let out: Box<[f32]> = (0..n_freqs)
        .map(|f| {
            let power: f32 = specs.iter().map(|s| (s[f] * wnorm).norm_sqr()).sum();
            (power / n_ch).max(1e-10).log10() * 10.0
        })
        .collect();
    let _ = sender.send(out);
}

struct Worker {
    model: DpdfNet,
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
    stats: Arc<StreamStats>,
    delay_ms: Arc<AtomicU32>,
    noise: NoiseControls,
}

impl Worker {
    fn run(mut self) {
        let m = &mut self.model;
        let (ch, hop, sr) = (m.ch, m.hop_size, m.sr);
        let wnorm = 2.0 * hop as f32 / (m.win_len * m.win_len) as f32;
        let mut inframe = vec![0.0f32; ch * hop];
        let mut outframe = vec![0.0f32; ch * hop];

        let (mut input_resampler, n_in) = if self.input_sr != sr {
            let r = FftFixedOut::<f32>::new(self.input_sr, sr, hop, 1, ch)
                .expect("Failed to init input resampler");
            let n_in = r.input_frames_max();
            let buf = r.input_buffer_allocate(true);
            (Some((r, buf)), n_in)
        } else {
            (None, hop)
        };
        let (mut output_resampler, n_out) = if self.output_sr != sr {
            let r = FftFixedIn::<f32>::new(sr, self.output_sr, hop, 1, ch)
                .expect("Failed to init output resampler");
            let n_out = r.output_frames_max();
            let buf = r.output_buffer_allocate(true);
            (Some((r, buf)), n_out)
        } else {
            (None, hop)
        };
        let mut interleaved_in = vec![0.0; n_in * ch];
        let mut interleaved_out = vec![0.0; n_out * ch];
        let mut resampled_in = vec![vec![0.0; hop]; ch];
        let mut smoothed_delay_s = 0.0f32;
        let mut noise = NoiseInjector::new(self.noise.clone(), sr, ch);

        // Transient smoother, same tuning as the DeepFilterNet pipeline.
        let per_sample_coeff = |tau_s: f32| (-1.0 / (tau_s * self.output_sr as f32)).exp();
        let env_attack = per_sample_coeff(0.030);
        let env_release = per_sample_coeff(0.300);
        let gain_attack = per_sample_coeff(0.003);
        let gain_release = per_sample_coeff(0.080);
        const SPIKE_THRESH_DB: f32 = 6.0;
        let mut env: f32 = -1.0;
        let mut gain: f32 = 1.0;

        log::info!(
            "DPDFNet worker started (input {} Hz, output {} Hz, model {} Hz)",
            self.input_sr,
            self.output_sr,
            sr
        );
        let mut frames_since_report = 0usize;
        let mut dropped_frames = 0usize;
        while !self.should_stop.load(Ordering::Relaxed) && !self.device_lost.load(Ordering::Relaxed)
        {
            if let Some(r_opt) = self.r_opt.as_ref() {
                while let Ok((c, v)) = r_opt.try_recv() {
                    if c == DfControl::AttenLim {
                        m.set_atten_lim(v);
                    }
                }
            }
            let queued_in = self.rb_in.len();
            if queued_in < n_in * ch {
                sleep(Duration::from_millis(1));
                continue;
            }
            // Fell behind (e.g. a stall): skip old input instead of staying late.
            if queued_in > 3 * n_in * ch {
                let excess = (queued_in / ch - n_in) * ch;
                self.rb_in.skip(excess);
                dropped_frames += 1;
            }
            let n = self.rb_in.pop_slice(&mut interleaved_in);
            debug_assert_eq!(n, interleaved_in.len());
            if let Some((ref mut r, ref mut buf)) = input_resampler.as_mut() {
                for (frame, samples) in interleaved_in.chunks_exact(ch).enumerate() {
                    for c in 0..ch {
                        buf[c][frame] = samples[c];
                    }
                }
                r.process_into_buffer(buf, &mut resampled_in, None).unwrap();
                for c in 0..ch {
                    inframe[c * hop..(c + 1) * hop].copy_from_slice(&resampled_in[c][..hop]);
                }
            } else {
                for (frame, samples) in interleaved_in.chunks_exact(ch).enumerate() {
                    for c in 0..ch {
                        inframe[c * hop + frame] = samples[c];
                    }
                }
            }

            noise.process_input(&mut inframe);
            let result = m.process(&inframe, &mut outframe);
            if let Err(e) = result {
                log::error!("DPDFNet processing failed: {e:#}");
                outframe.fill(0.0);
            }
            noise.process_output(&mut outframe);

            if let Some((ref mut r, ref mut buf)) = output_resampler.as_mut() {
                let rows: Vec<&[f32]> = outframe.chunks_exact(hop).collect();
                r.process_into_buffer(&rows, buf, None).unwrap();
                for frame in 0..n_out {
                    for c in 0..ch {
                        interleaved_out[frame * ch + c] = buf[c][frame];
                    }
                }
            } else {
                for frame in 0..hop {
                    for c in 0..ch {
                        interleaved_out[frame * ch + c] = outframe[c * hop + frame];
                    }
                }
            }

            for frame in interleaved_out.chunks_mut(ch) {
                let level = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
                if env < 0.0 {
                    env = level.max(1e-8);
                }
                env = if level > env {
                    env * env_attack + level * (1.0 - env_attack)
                } else {
                    env * env_release + level * (1.0 - env_release)
                };
                let level_db = 20.0 * (level.max(1e-8) / env.max(1e-8)).log10();
                let target_gain = if level_db > SPIKE_THRESH_DB {
                    10f32.powf(-(level_db - SPIKE_THRESH_DB) / 20.0)
                } else {
                    1.0
                };
                let coeff = if target_gain < gain {
                    gain_attack
                } else {
                    gain_release
                };
                gain = gain * coeff + target_gain * (1.0 - coeff);
                if self.smoother_enabled.load(Ordering::Relaxed) {
                    for s in frame.iter_mut() {
                        *s *= gain;
                    }
                }
            }
            // Apply Windows master volume (and mute) to every output sample.
            let vol = f32::from_bits(self.shared_volume.load(Ordering::Relaxed));
            for s in interleaved_out.iter_mut() {
                *s *= vol;
            }
            // Drop this frame rather than queue more than the device needs.
            if self.rb_out.len() + interleaved_out.len() > max_output_queue(&self.stats, n_out, ch)
            {
                dropped_frames += 1;
            } else {
                self.rb_out.push_slice(&interleaved_out);
                self.rb_out.sync();
            }
            // Delay estimate for the UI: model lag, one frame of buffering, and
            // the audio queued on both sides.
            let delay_s = (m.delay_samples() + hop) as f32 / sr as f32
                + self.rb_in.len() as f32 / ch as f32 / self.input_sr as f32
                + self.rb_out.len() as f32 / ch as f32 / self.output_sr as f32;
            smoothed_delay_s = if smoothed_delay_s == 0.0 {
                delay_s
            } else {
                0.95 * smoothed_delay_s + 0.05 * delay_s
            };
            self.delay_ms
                .store((smoothed_delay_s * 1000.0) as u32, Ordering::Relaxed);

            frames_since_report += 1;
            if frames_since_report * hop >= sr * 2 && log::log_enabled!(log::Level::Debug) {
                frames_since_report = 0;
                let ms =
                    |samples: usize, rate: usize| samples as f32 / ch as f32 * 1000.0 / rate as f32;
                log::debug!(
                    "DPDFNet queued audio: input {:.1} ms, output {:.1} ms \
                     (output callback {} frames; underruns {}, input overflows {}, dropped frames {})",
                    ms(self.rb_in.len(), self.input_sr),
                    ms(self.rb_out.len(), self.output_sr),
                    self.stats.out_callback_frames.load(Ordering::Relaxed),
                    self.stats.underruns.load(Ordering::Relaxed),
                    self.stats.input_overflows.load(Ordering::Relaxed),
                    dropped_frames
                );
            }

            if let Some((ref s_noisy, ref s_enh)) = self.s_spec {
                let noisy: Vec<_> = (0..ch).map(|c| m.spec_noisy(c)).collect();
                let enh: Vec<_> = (0..ch).map(|c| m.spec_enh(c)).collect();
                push_spec(&noisy, wnorm, s_noisy);
                push_spec(&enh, wnorm, s_enh);
            }
        }
        log::info!("DPDFNet worker stopped");
    }
}

pub struct DpdfNetCapture {
    pub sr: usize,
    pub frame_size: usize,
    pub freq_size: usize,
    should_stop: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
    input_stream: Option<Stream>,
    output_stream: Option<Stream>,
    /// Current estimate of the input-to-output delay.
    pub delay_ms: Arc<AtomicU32>,
}

impl DpdfNetCapture {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_kind: DpdfModelKind,
        input_device: Option<String>,
        output_device: Option<String>,
        s_noisy: Option<SendSpec>,
        s_enh: Option<SendSpec>,
        r_opt: Option<RecvControl>,
        s_device_event: Option<SendDeviceEvent>,
        shared_volume: Arc<AtomicU32>,
        smoother_enabled: Arc<AtomicBool>,
        noise: NoiseControls,
    ) -> Result<Self> {
        let ch = PROCESS_CHANNELS;
        let mut model = DpdfNet::new(model_kind, ch)?;
        let (sr, frame_size, freq_size) = (model.sr, model.hop_size, model.n_freqs);

        // Warm up so the first real frame doesn't pay session initialisation.
        let warm = vec![0.0f32; ch * frame_size];
        let mut warm_out = warm.clone();
        model.process(&warm, &mut warm_out)?;
        model.reset();

        let in_dev = find_device(input_device, Direction::Input)?;
        let out_dev = find_device(output_device, Direction::Output)?;
        let in_cfg = stream_config(&in_dev, sr as u32, frame_size, Direction::Input)?;
        let out_cfg = stream_config(&out_dev, sr as u32, frame_size, Direction::Output)?;

        let (in_prod, in_cons) = HeapRb::<f32>::new(frame_size * ch * 100).split();
        let (out_prod, out_cons) = HeapRb::<f32>::new(frame_size * ch * 100).split();

        let should_stop = Arc::new(AtomicBool::new(false));
        let device_lost = Arc::new(AtomicBool::new(false));
        let delay_ms = Arc::new(AtomicU32::new(0));
        let stats = Arc::new(StreamStats::default());
        let worker = Worker {
            model,
            rb_in: in_cons,
            rb_out: out_prod.into_postponed(),
            input_sr: in_cfg.sample_rate.0 as usize,
            output_sr: out_cfg.sample_rate.0 as usize,
            should_stop: should_stop.clone(),
            device_lost: device_lost.clone(),
            s_spec: s_noisy.zip(s_enh),
            r_opt,
            shared_volume,
            smoother_enabled,
            stats: stats.clone(),
            delay_ms: delay_ms.clone(),
            noise,
        };
        let worker_handle = Some(
            thread::Builder::new()
                .name("dpdfnet-worker".into())
                .spawn(move || worker.run())?,
        );

        // Output first: if input started first, audio would queue up while the
        // output device opens, and that delay would stay for the whole session.
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
        log::info!("DPDFNet capture started with {}", model_kind.label());

        Ok(Self {
            sr,
            frame_size,
            freq_size,
            should_stop,
            worker_handle,
            input_stream: Some(input_stream),
            output_stream: Some(output_stream),
            delay_ms,
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
            log::info!("Joining DPDFNet worker");
            self.should_stop.store(true, Ordering::Relaxed);
            let _ = h.join();
        }
        Ok(())
    }
}

impl Drop for DpdfNetCapture {
    fn drop(&mut self) {
        let _ = self.should_stop();
    }
}
