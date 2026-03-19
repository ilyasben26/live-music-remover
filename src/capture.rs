use std::env;
use std::fmt::Display;
use std::io::{self, stdout, Write};
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Once,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Device, SampleRate, Stream, StreamConfig, SupportedStreamConfigRange};
use crossbeam_channel::{unbounded, Receiver, Sender};
use df::{tract::*, Complex32};
use ndarray::prelude::*;
use ringbuf::{producer::PostponedProducer, Consumer, HeapRb, SharedRb};
use rubato::{FftFixedIn, FftFixedOut, Resampler};

pub type RbProd = PostponedProducer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>;
pub type RbCons = Consumer<f32, Arc<SharedRb<f32, Vec<MaybeUninit<f32>>>>>;
pub type SendLsnr = Sender<f32>;
pub type RecvLsnr = Receiver<f32>;
pub type SendSpec = Sender<Box<[f32]>>;
pub type RecvSpec = Receiver<Box<[f32]>>;
pub type SendControl = Sender<(DfControl, f32)>;
pub type RecvControl = Receiver<(DfControl, f32)>;

pub(crate) static INIT_LOGGER: Once = Once::new();
pub(crate) static mut MODEL_PATH: Option<PathBuf> = None;
static mut MODEL: Option<DfTract> = None;

const SAMPLE_FORMAT: cpal::SampleFormat = cpal::SampleFormat::F32;
const PROCESS_CHANNELS: usize = 2;

pub struct AudioSink {
    stream: Option<Stream>,
    config: StreamConfig,
    device: Device,
}
pub struct AudioSource {
    stream: Option<Stream>,
    config: StreamConfig,
    device: Device,
}

#[derive(PartialEq)]
pub enum DfControl {
    AttenLim,
    PostFilterBeta,
    MinThreshDb,
    MaxErbThreshDb,
    MaxDfThreshDb,
}

/// Initialize DF model and returns sample rate, frame size, and number of frequency bins
fn init_df(model_path: Option<PathBuf>, channels: usize) -> (usize, usize, usize) {
    unsafe {
        if let Some(m) = MODEL.as_ref() {
            if m.ch == channels {
                return (m.sr, m.hop_size, m.n_freqs);
            }
        }
    }
    // let df_params = DfParams::default();
    let df_params = if let Some(path) = model_path {
        log::debug!("Not using default DF model, loading from path: {:?}", path);
        DfParams::new(path).expect("Failed to read DF model")
    } else {
        log::debug!("Using default DF model");
        DfParams::default()
    };
    let r_params = RuntimeParams::default_with_ch(channels);
    let df = DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime");
    let (sr, frame_size, freq_size) = (df.sr, df.hop_size, df.n_freqs);
    unsafe { MODEL = Some(df) };
    (sr, frame_size, freq_size)
}

unsafe fn get_frame_size() -> usize {
    let df = MODEL.clone().unwrap();
    df.hop_size
}

#[derive(Clone, Copy)]
enum StreamDirection {
    Input,
    Output,
}
impl Display for StreamDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamDirection::Input => write!(f, "input"),
            StreamDirection::Output => write!(f, "output"),
        }
    }
}

fn get_all_configs(device: &Device, direction: StreamDirection) -> Vec<SupportedStreamConfigRange> {
    match direction {
        StreamDirection::Input => device
            .supported_input_configs()
            .expect("Failed to get input configs")
            .collect::<Vec<SupportedStreamConfigRange>>(),
        StreamDirection::Output => device
            .supported_output_configs()
            .expect("Failed to get output configs")
            .collect::<Vec<SupportedStreamConfigRange>>(),
    }
}

fn get_stream_config(
    device: &Device,
    sample_rate: u32,
    direction: StreamDirection,
) -> Option<StreamConfig> {
    let mut configs = Vec::new();
    let all_configs = get_all_configs(device, direction);
    for c in all_configs.iter() {
        if c.channels() == PROCESS_CHANNELS as u16 && c.sample_format() == SAMPLE_FORMAT {
            log::debug!("Found audio {} config: {:?}", direction, &c);
            configs.push(*c);
        }
    }
    // Prefer mono as a fallback when stereo is not available.
    for c in all_configs.iter() {
        if c.channels() == 1 && c.sample_format() == SAMPLE_FORMAT {
            log::debug!("Found audio {} config: {:?}", direction, &c);
            configs.push(*c);
        }
    }
    // Further add multi-channel configs as a final fallback.
    for c in all_configs.iter() {
        if c.channels() > PROCESS_CHANNELS as u16 && c.sample_format() == SAMPLE_FORMAT {
            log::debug!("Found audio {} config: {:?}", direction, &c);
            configs.push(*c);
        }
    }
    assert!(
        !configs.is_empty(),
        "No suitable audio {} config found.",
        direction
    );
    let sr = SampleRate(sample_rate);
    for c in configs.iter() {
        if sr >= c.min_sample_rate() && sr <= c.max_sample_rate() {
            let mut c: StreamConfig = (*c).with_sample_rate(sr).into();
            c.buffer_size = BufferSize::Fixed(unsafe { get_frame_size() } as u32);
            return Some(c);
        }
    }

    if let Some(c) = configs.first() {
        let mut c: StreamConfig = (*c).with_max_sample_rate().into();
        c.buffer_size =
            BufferSize::Fixed(unsafe { get_frame_size() } as u32 * c.sample_rate.0 / sample_rate);
        log::warn!("Using best matching config {:?}", c);
        return Some(c);
    }
    None
}

fn remap_interleaved_channels(input: &[f32], input_ch: usize, output: &mut [f32], output_ch: usize) {
    debug_assert_eq!(output.len(), input.len() / input_ch * output_ch);
    let n_frames = input.len() / input_ch;
    for frame in 0..n_frames {
        let in_frame = &input[frame * input_ch..(frame + 1) * input_ch];
        let out_frame = &mut output[frame * output_ch..(frame + 1) * output_ch];
        for ch in 0..output_ch {
            out_frame[ch] = if ch < input_ch {
                in_frame[ch]
            } else {
                in_frame[0]
            };
        }
    }
}

fn interleaved_to_planar(input: &[f32], channels: usize, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let n_frames = input.len() / channels;
    for frame in 0..n_frames {
        for ch in 0..channels {
            output[ch * n_frames + frame] = input[frame * channels + ch];
        }
    }
}

fn planar_to_interleaved(input: &[f32], channels: usize, output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let n_frames = input.len() / channels;
    for frame in 0..n_frames {
        for ch in 0..channels {
            output[frame * channels + ch] = input[ch * n_frames + frame];
        }
    }
}

impl AudioSink {
    fn new(sample_rate: u32, device_str: Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let mut device = host.default_output_device().expect("no output device available");
        if let Some(device_str) = device_str {
            for avail_dev in host.output_devices()? {
                if avail_dev.name()?.to_lowercase().contains(&device_str.to_lowercase()) {
                    device = avail_dev
                }
            }
        }
        let config = get_stream_config(&device, sample_rate, StreamDirection::Output)
            .expect("No suitable audio output config found.");

        log::info!("Selected audio output device: {}", device.name()?);
        log::info!("Selected audio output config: {:?}", config);

        Ok(Self {
            stream: None,
            config,
            device,
        })
    }
    fn start(&mut self, mut rb: RbCons, model_ch: usize) -> Result<()> {
        let output_ch = self.config.channels as usize;
        let stream = self.device.build_output_stream(
            &self.config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                if output_ch == model_ch {
                    let mut n = 0;
                    while n < data.len() {
                        n += rb.pop_slice(&mut data[n..]);
                    }
                    debug_assert_eq!(n, data.len());
                } else {
                    let n_frames = data.len() / output_ch;
                    let mut rb_data = vec![0.0; n_frames * model_ch];
                    let mut n = 0;
                    while n < rb_data.len() {
                        n += rb.pop_slice(&mut rb_data[n..]);
                    }
                    debug_assert_eq!(n, rb_data.len());
                    remap_interleaved_channels(&rb_data, model_ch, data, output_ch);
                }
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!(
                        "Returning data to audio sink with len: {}, rms: {}",
                        data.len() / output_ch,
                        df::rms(data.iter())
                    );
                }
            },
            move |err| log::error!("Error during audio output {:?}", err),
            None, // None=blocking, Some(Duration)=timeout
        )?;
        stream.play()?;
        log::info!("Starting playback stream on device {}", self.device.name()?);
        self.stream = Some(stream);
        Ok(())
    }
    fn sr(&self) -> u32 {
        self.config.sample_rate.0
    }
    fn pause(&mut self) -> Result<()> {
        if let Some(s) = self.stream.as_mut() {
            s.pause()?;
        }
        Ok(())
    }
}

impl AudioSource {
    fn new(sample_rate: u32, device_str: Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let mut device = host.default_input_device().expect("no output device available");
        if let Some(device_str) = device_str {
            for avail_dev in host.input_devices()? {
                if avail_dev.name()?.to_lowercase().contains(&device_str.to_lowercase()) {
                    device = avail_dev
                }
            }
        }
        let config = get_stream_config(&device, sample_rate, StreamDirection::Input)
            .expect("No suitable audio input config found.");

        log::info!("Selected audio input device: {}", device.name()?);
        log::info!("Selected audio input config: {:?}", config);

        Ok(Self {
            stream: None,
            config,
            device,
        })
    }
    fn start(&mut self, mut rb: RbProd, model_ch: usize) -> Result<()> {
        let input_ch = self.config.channels as usize;
        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let n_frames = data.len() / input_ch;
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!(
                        "Got data from audio source with len: {}, rms: {}",
                        n_frames,
                        df::rms(data.iter())
                    );
                }
                if input_ch == model_ch {
                    let mut n = 0;
                    while n < data.len() {
                        n += rb.push_slice(&data[n..]);
                    }
                    debug_assert_eq!(n, data.len());
                } else {
                    let mut mapped = vec![0.0; n_frames * model_ch];
                    remap_interleaved_channels(data, input_ch, &mut mapped, model_ch);
                    let mut n = 0;
                    while n < mapped.len() {
                        n += rb.push_slice(&mapped[n..]);
                    }
                    debug_assert_eq!(n, mapped.len());
                }
                rb.sync();
            },
            move |err| log::error!("Error during audio output {:?}", err),
            None, // None=blocking, Some(Duration)=timeout
        )?;
        log::info!("Starting capture stream on device {}", self.device.name()?);
        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }
    fn sr(&self) -> u32 {
        self.config.sample_rate.0
    }
    fn pause(&mut self) -> Result<()> {
        if let Some(s) = self.stream.as_mut() {
            s.pause()?;
        }
        Ok(())
    }
}

pub(crate) struct AtomicControls {
    has_init: Arc<AtomicBool>,
    should_stop: Arc<AtomicBool>,
}
impl AtomicControls {
    pub fn into_inner(self) -> (Arc<AtomicBool>, Arc<AtomicBool>) {
        (self.has_init, self.should_stop)
    }
}
pub(crate) struct GuiCom {
    pub s_lsnr: Option<SendLsnr>,
    pub s_spec: Option<(SendSpec, SendSpec)>,
    pub r_opt: Option<RecvControl>,
}
impl GuiCom {
    pub fn into_inner(
        self,
    ) -> (
        Option<SendLsnr>,
        Option<(SendSpec, SendSpec)>,
        Option<RecvControl>,
    ) {
        (self.s_lsnr, self.s_spec, self.r_opt)
    }
}

fn get_worker_fn(
    mut rb_in: RbCons,
    mut rb_out: RbProd,
    input_sr: usize,
    output_sr: usize,
    controls: AtomicControls,
    df_com: Option<GuiCom>,
) -> impl FnMut() {
    let (has_init, should_stop) = controls.into_inner();
    let (mut s_lsnr, mut s_spec, mut r_opt) = if let Some(df_com) = df_com {
        df_com.into_inner()
    } else {
        (None, None, None)
    };
    move || {
        let mut df = unsafe { MODEL.clone().unwrap() };
        let ch = df.ch;
        let mut inframe = Array2::zeros((df.ch, df.hop_size));
        let mut outframe = inframe.clone();
        df.process(inframe.view(), outframe.view_mut())
            .expect("Failed to run DeepFilterNet");
        has_init.store(true, Ordering::Relaxed);
        log::info!("Worker init");
        let (mut input_resampler, n_in) = if input_sr != df.sr {
            let r = FftFixedOut::<f32>::new(input_sr, df.sr, df.hop_size, 1, ch)
                .expect("Failed to init input resampler");
            let n_in = r.input_frames_max();
            let buf = r.input_buffer_allocate(true);
            (Some((r, buf)), n_in)
        } else {
            (None, df.hop_size)
        };
        let (mut output_resampler, n_out) = if output_sr != df.sr {
            let r = FftFixedIn::<f32>::new(df.sr, output_sr, df.hop_size, 1, ch)
                .expect("Failed to init output resampler");
            let n_out = r.output_frames_max();
            let buf = r.output_buffer_allocate(true);
            (Some((r, buf)), n_out)
        } else {
            (None, df.hop_size)
        };
        let mut interleaved_in = vec![0.0; n_in * ch];
        let mut interleaved_out = vec![0.0; n_out * ch];
        let mut resampled_in = vec![vec![0.0; df.hop_size]; ch];
        while !should_stop.load(Ordering::Relaxed) {
            if rb_in.len() < n_in * ch {
                // Sleep for half a hop size
                sleep(Duration::from_secs_f32(
                    df.hop_size as f32 / df.sr as f32 / 2.,
                ));
                continue;
            }
            if let Some((ref mut r, ref mut buf)) = input_resampler.as_mut() {
                let n = rb_in.pop_slice(&mut interleaved_in);
                debug_assert_eq!(n, interleaved_in.len());
                debug_assert_eq!(n_in, r.input_frames_next());
                for (frame, frame_samples) in interleaved_in.chunks_exact(ch).enumerate() {
                    for c in 0..ch {
                        buf[c][frame] = frame_samples[c];
                    }
                }
                r.process_into_buffer(buf, &mut resampled_in, None).unwrap();
                for c in 0..ch {
                    for i in 0..df.hop_size {
                        inframe[(c, i)] = resampled_in[c][i];
                    }
                }
            } else {
                let n = rb_in.pop_slice(&mut interleaved_in);
                debug_assert_eq!(n, interleaved_in.len());
                interleaved_to_planar(&interleaved_in, ch, inframe.as_slice_mut().unwrap());
            }
            let lsnr = df
                .process(inframe.view(), outframe.view_mut())
                .expect("Failed to run DeepFilterNet");
            let mut n = 0;
            if let Some((ref mut r, ref mut buf)) = output_resampler.as_mut() {
                let out_rows = (0..ch)
                    .map(|c| outframe.row(c).to_vec())
                    .collect::<Vec<_>>();
                let out_rows = out_rows.iter().map(|row| row.as_slice()).collect::<Vec<_>>();
                r.process_into_buffer(&out_rows, buf, None).unwrap();
                for frame in 0..n_out {
                    for c in 0..ch {
                        interleaved_out[frame * ch + c] = buf[c][frame];
                    }
                }
                while n < interleaved_out.len() {
                    n += rb_out.push_slice(&interleaved_out[n..]);
                }
            } else {
                planar_to_interleaved(outframe.as_slice().unwrap(), ch, &mut interleaved_out);
                while n < interleaved_out.len() {
                    n += rb_out.push_slice(&interleaved_out[n..]);
                }
            }
            debug_assert_eq!(n, n_out * ch);
            rb_out.sync();
            if let Some(ref mut s_lsnr) = s_lsnr.as_mut() {
                s_lsnr.send(lsnr).expect("Failed to send to LSNR rb");
            }
            if let Some((ref mut s_noisy, ref mut s_enh)) = s_spec.as_mut() {
                push_spec(df.get_spec_noisy(), s_noisy);
                push_spec(df.get_spec_enh(), s_enh);
            }
            if let Some(ref mut r_opt) = r_opt.as_mut() {
                while let Ok((c, v)) = r_opt.try_recv() {
                    match c {
                        DfControl::AttenLim => df.set_atten_lim(v),
                        DfControl::PostFilterBeta => df.set_pf_beta(v),
                        DfControl::MinThreshDb => df.min_db_thresh = v,
                        DfControl::MaxErbThreshDb => df.max_db_erb_thresh = v,
                        DfControl::MaxDfThreshDb => df.max_db_df_thresh = v,
                    }
                }
            }
        }
    }
}

fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
    let n_ch = spec.len_of(Axis(0));
    let n_freqs = spec.len_of(Axis(1));
    let mut out = vec![0.0; n_freqs];
    for f in 0..n_freqs {
        let mut power = 0.0;
        for ch in 0..n_ch {
            power += spec[(ch, f)].norm_sqr();
        }
        out[f] = (power / n_ch as f32).max(1e-10).log10() * 10.0;
    }
    sender.send(out.into_boxed_slice()).expect("Failed to send spectrogram")
}

pub fn log_format(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> io::Result<()> {
    let ts = buf.timestamp_millis();
    let module = record.module_path().unwrap_or("").to_string();
    let level_style = buf.default_level_style(log::Level::Info);

    writeln!(
        buf,
        "{} | {} | {} {}",
        ts,
        level_style.value(record.level()),
        module,
        record.args()
    )
}

pub struct DeepFilterCapture {
    pub sr: usize,
    pub frame_size: usize,
    pub freq_size: usize,
    should_stop: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
    source: AudioSource,
    sink: AudioSink,
}

impl Default for DeepFilterCapture {
    fn default() -> Self {
        DeepFilterCapture::new(None, None, None, None, None)
            .expect("Error during DeepFilterCapture initialization")
    }
}
impl DeepFilterCapture {
    pub fn new(
        model_path: Option<PathBuf>,
        s_lsnr: Option<SendLsnr>,
        s_noisy: Option<SendSpec>,
        s_enh: Option<SendSpec>,
        r_opt: Option<RecvControl>,
    ) -> Result<Self> {
        let ch = PROCESS_CHANNELS;
        let (sr, frame_size, freq_size) = init_df(model_path, ch);
        let in_rb = HeapRb::<f32>::new(frame_size * ch * 100);
        let out_rb = HeapRb::<f32>::new(frame_size * ch * 100);
        let (in_prod, in_cons) = in_rb.split();
        let (out_prod, out_cons) = out_rb.split();
        let in_prod = in_prod.into_postponed();
        let out_prod = out_prod.into_postponed();

        let input_device: Option<String> = Some("CABLE Input".to_string());
        let output_device: Option<String> = Some("Headphones".to_string());

        let mut source = AudioSource::new(sr as u32, input_device)?;
        let mut sink = AudioSink::new(sr as u32, output_device)?;
        let should_stop = Arc::new(AtomicBool::new(false));
        let has_init = Arc::new(AtomicBool::new(false));
        let s_spec = match (s_noisy, s_enh) {
            (Some(n), Some(e)) => Some((n, e)),
            _ => None,
        };
        let controls = AtomicControls {
            has_init: has_init.clone(),
            should_stop: should_stop.clone(),
        };
        let df_com = GuiCom {
            s_lsnr,
            s_spec,
            r_opt,
        };
        let worker_handle = Some(thread::spawn(get_worker_fn(
            in_cons,
            out_prod,
            source.sr() as usize,
            sink.sr() as usize,
            controls,
            Some(df_com),
        )));
        while !has_init.load(Ordering::Relaxed) {
            sleep(Duration::from_secs_f32(0.01));
        }
        log::info!("DeepFilter Capture init");
        source.start(in_prod, ch)?;
        sink.start(out_cons, ch)?;

        Ok(Self {
            sr,
            frame_size,
            freq_size,
            should_stop,
            worker_handle,
            source,
            sink,
        })
    }

    pub fn should_stop(&mut self) -> Result<()> {
        self.sink.pause()?;
        self.source.pause()?;
        if let Some(h) = self.worker_handle.take() {
            log::info!("Joining DF Worker");
            self.should_stop.swap(true, Ordering::Relaxed);
            h.join().expect("Error during DF worker join");
        }
        Ok(())
    }
}

#[allow(unused)]
#[allow(unknown_lints)] // assigning_clones is clippy nightly only
#[allow(clippy::assigning_clones)]
pub fn main() -> Result<()> {
    INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
            .filter_module("tract_onnx", log::LevelFilter::Error)
            .filter_module("tract_core", log::LevelFilter::Error)
            .filter_module("tract_hir", log::LevelFilter::Error)
            .filter_module("tract_linalg", log::LevelFilter::Error)
            .format(log_format)
            .init();
    });

    let (lsnr_prod, mut lsnr_cons) = unbounded();
    let mut model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
    unsafe {
        if model_path.is_none() && MODEL_PATH.is_some() {
            model_path = MODEL_PATH.clone()
        }
    }
    if let Some(p) = model_path.as_ref() {
        log::info!("Running with model '{:?}'", p);
    }
    let _c = DeepFilterCapture::new(model_path, Some(lsnr_prod), None, None, None);

    loop {
        sleep(Duration::from_millis(200));
        while let Ok(lsnr) = lsnr_cons.try_recv() {
            print!("\rCurrent SNR: {:>5.1} dB{esc}[1;", lsnr, esc = 27 as char);
        }
        stdout().flush().unwrap();
    }
}
