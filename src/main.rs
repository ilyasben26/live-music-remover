#![cfg_attr(
    all(target_os = "windows", not(feature = "dev-console")),
    windows_subsystem = "windows"
)]

use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use crate::volume::SystemVolume;
use clap::Parser;
use crossbeam_channel::{unbounded, Receiver};
use eframe::egui;
use image_rs::{Rgba, RgbaImage};

mod capture;
mod cmap;
mod devices;
mod dpdfnet;
mod dpdfnet_capture;
mod notify_update;
mod volume;
mod win_notification;
use capture::{ModelKind, *};
use dpdfnet::DpdfModelKind;
use dpdfnet_capture::DpdfNetCapture;
use egui_router::{EguiRouter, Route, TransitionConfig};

use crate::devices::{
    device_exists, find_cable, get_input_devices, get_output_devices, set_output_device,
};

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Logging verbosity
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
        help = "Increase logging verbosity with multiple `-vv`",
    )]
    verbose: u8,
}

fn load_icon() -> egui::IconData {
    let icon_bytes = include_bytes!("../assets/logo.png");
    let img = image_rs::load_from_memory(icon_bytes)
        .expect("Failed to load icon")
        .into_rgba8();
    let (width, height) = img.dimensions();
    egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    }
}

pub fn main() -> eframe::Result<()> {
    let args = Args::parse();
    let level = match args.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    let tract_level = match args.verbose {
        0..=3 => log::LevelFilter::Error,
        4 => log::LevelFilter::Info,
        5 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    capture::INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default())
            .filter_level(level)
            .filter_module("tract_onnx", tract_level)
            .filter_module("tract_hir", tract_level)
            .filter_module("tract_core", tract_level)
            .filter_module("tract_linalg", tract_level)
            .filter_module("eframe", log::LevelFilter::Error)
            .filter_module("egui_wgpu", log::LevelFilter::Error)
            .filter_module("wgpu_core", log::LevelFilter::Error)
            .filter_module("wgpu_hal", log::LevelFilter::Error)
            .filter_module("naga", log::LevelFilter::Error)
            .format(capture::log_format)
            .init();
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Live Music Remover")
            .with_inner_size([1200.0, 750.0])
            .with_min_inner_size([900.0, 500.0])
            .with_icon(load_icon()),
        ..Default::default()
    };

    log::debug!("CARGO_PKG_VERSION: {}", env!("CARGO_PKG_VERSION"));
    log::debug!("CARGO_PKG_NAME: {}", env!("CARGO_PKG_NAME"));

    eframe::run_native(
        "Live Music Remover",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_zoom_factor(1.2);
            Ok(Box::new(LiveMusicRemover::new()))
        }),
    )
}

/// A model and the pipeline that runs it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelChoice {
    DeepFilter(ModelKind),
    DpdfNet(DpdfModelKind),
}

impl ModelChoice {
    const ALL: [ModelChoice; 4] = [
        ModelChoice::DeepFilter(ModelKind::Standard),
        ModelChoice::DeepFilter(ModelKind::LowLatency),
        ModelChoice::DpdfNet(DpdfModelKind::DpdfNet2_48kHr),
        ModelChoice::DpdfNet(DpdfModelKind::DpdfNet8_48kHr),
    ];

    fn label(self) -> &'static str {
        match self {
            ModelChoice::DeepFilter(k) => k.label(),
            ModelChoice::DpdfNet(k) => k.label(),
        }
    }

    fn is_deep_filter(self) -> bool {
        matches!(self, ModelChoice::DeepFilter(_))
    }
}

enum CaptureWorker {
    DeepFilter(DeepFilterCapture),
    DpdfNet(DpdfNetCapture),
}

impl CaptureWorker {
    fn should_stop(&mut self) -> anyhow::Result<()> {
        match self {
            CaptureWorker::DeepFilter(w) => w.should_stop(),
            CaptureWorker::DpdfNet(w) => w.should_stop(),
        }
    }
}

#[derive(PartialEq, Clone)]
enum Page {
    Main,
    About,
    Help,
}

struct LiveMusicRemover {
    freq_axis_scale: f32,
    df_worker: Option<CaptureWorker>,
    lsnr: f32,
    atten_lim: f32,
    post_filter_beta: f32,
    min_threshdb: f32,
    max_erbthreshdb: f32,
    max_dfthreshdb: f32,
    spec_noisy: Option<SpecImage>,
    spec_enh: Option<SpecImage>,
    noisy_texture: Option<egui::TextureHandle>,
    enh_texture: Option<egui::TextureHandle>,
    logo_texture: Option<egui::TextureHandle>,
    r_lsnr: RecvLsnr,
    r_noisy: RecvSpec,
    r_enh: RecvSpec,
    s_controls: SendControl,
    input_device: Option<String>,
    output_device: Option<String>,
    available_input_devices: Vec<String>,
    available_output_devices: Vec<String>,
    r_device_event: Option<Receiver<DeviceEvent>>,
    shared_volume: Arc<AtomicU32>,
    system_volume: f32,
    smoother_enabled: Arc<AtomicBool>,
    selected_model: ModelChoice,
    dark_mode: bool,
    last_dark_mode: bool,
    router: Option<EguiRouter<LiveMusicRemover>>,
    current_page: Page,
    update_info: Arc<Mutex<Option<(String, String)>>>,
}

struct SpecImage {
    im: RgbaImage,
    n_frames: u32,
    n_freqs: u32,
    vmin: f32,
    vmax: f32,
}

impl SpecImage {
    fn new(n_frames: u32, n_freqs: u32, vmin: f32, vmax: f32) -> Self {
        Self {
            // Store image transposed so we can iterate over rows quickly
            im: RgbaImage::new(n_freqs, n_frames),
            n_frames,
            n_freqs,
            vmin,
            vmax,
        }
    }
    fn w(&self) -> usize {
        self.n_frames as usize
    }
    fn h(&self) -> usize {
        self.n_freqs as usize
    }
    fn update<I>(&mut self, specs: I, mut n_specs: usize)
    where
        I: Iterator<Item = Box<[f32]>>,
    {
        if n_specs == 0 {
            return;
        }
        if n_specs >= self.n_frames as usize {
            n_specs = self.n_frames as usize - 1;
        }
        for (spec, im_row) in specs.take(n_specs).zip(self.im.rows_mut()) {
            for (s, x) in spec.iter().zip(im_row) {
                let v = (s.min(self.vmax).max(self.vmin) - self.vmin) / (self.vmax - self.vmin);
                *x = Rgba(cmap::CMAP_INFERNO[(v * 255.) as usize]);
            }
        }
        let (w, h) = (self.w(), self.h());
        self.im.rotate_left((w - n_specs) * 4 * h);
    }
    fn to_color_image(&self, dark_mode: bool) -> egui::ColorImage {
        // Time on x-axis (left=old, right=new), frequency on y-axis (top=high, bottom=low)
        let nf = self.n_freqs as usize;
        let nt = self.n_frames as usize;
        let bg = if dark_mode {
            egui::Color32::BLACK
        } else {
            egui::Color32::from_rgba_unmultiplied(220, 220, 220, 255)
        };
        let mut pixels = vec![bg; nf * nt];
        for (t, row) in self.im.rows().enumerate() {
            for (f, p) in row.enumerate() {
                let out_x = t; // time: left=old, right=new
                let out_y = nf - 1 - f; // frequency: top=high, bottom=low
                if p[3] > 0 {
                    let color = if dark_mode {
                        egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3])
                    } else {
                        egui::Color32::from_rgba_unmultiplied(
                            220 - p[0],
                            220 - p[1],
                            220 - p[2],
                            p[3],
                        )
                    };
                    pixels[out_y * nt + out_x] = color;
                }
            }
        }
        egui::ColorImage {
            size: [nt, nf],
            pixels,
        }
    }
}

impl LiveMusicRemover {
    fn new() -> Self {
        let update_info: Arc<Mutex<Option<(String, String)>>> = Arc::new(Mutex::new(None));
        let update_info_thread = Arc::clone(&update_info);
        std::thread::spawn(move || notify_update::check_for_update(update_info_thread));

        let (_s_lsnr, r_lsnr) = unbounded();
        let (_s_noisy, r_noisy) = unbounded();
        let (_s_enh, r_enh) = unbounded();
        let (s_controls, _r_controls) = unbounded();
        let (_s_device_event, r_device_event) = unbounded();
        let shared_volume = Arc::new(AtomicU32::new(1.0f32.to_bits()));
        let sv = shared_volume.clone();
        std::thread::spawn(move || {
            let sys_vol = SystemVolume::new()
                .map_err(|e| log::warn!("Volume monitor: {e}"))
                .ok();
            loop {
                let vol = sys_vol
                    .as_ref()
                    .map(|v: &SystemVolume| v.get_scalar())
                    .unwrap_or(1.0);
                sv.store(vol.to_bits(), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });

        let available_input_devices = get_input_devices().unwrap_or_else(|e| {
            log::error!("Failed to get input devices: {}", e);
            vec![]
        });

        let available_output_devices = get_output_devices().unwrap_or_else(|e| {
            log::error!("Failed to get output devices: {}", e);
            vec![]
        });

        log::debug!("available_input_devices: {:?}", available_input_devices);
        log::debug!("available_output_devices: {:?}", available_output_devices);

        let input_device = match find_cable() {
            Ok(name) => {
                log::info!("Auto-selected input device: {name}");
                Some(name)
            }
            Err(e) => {
                log::warn!("Could not auto-select 'CABLE Input': {e}");
                None
            }
        };

        let output_device = match set_output_device() {
            Ok(name) => {
                log::info!("Auto-selected output device: {name}");
                Some(name)
            }
            Err(e) => {
                log::warn!("Could not auto-select output device: {e}");
                None
            }
        };

        let mut state = Self {
            df_worker: None,
            lsnr: 0.,
            atten_lim: 100.,
            post_filter_beta: 1.,
            min_threshdb: -15.,
            max_erbthreshdb: 35.,
            max_dfthreshdb: 35.,
            spec_noisy: None,
            spec_enh: None,
            noisy_texture: None,
            enh_texture: None,
            logo_texture: None,
            r_lsnr,
            r_noisy,
            r_enh,
            s_controls,
            input_device,
            output_device,
            available_input_devices,
            available_output_devices,
            r_device_event: Some(r_device_event),
            shared_volume,
            system_volume: 1.0,
            smoother_enabled: Arc::new(AtomicBool::new(true)),
            freq_axis_scale: 1.5,
            selected_model: ModelChoice::DeepFilter(ModelKind::default()),
            dark_mode: true,
            last_dark_mode: true,
            router: None,
            current_page: Page::Main,
            update_info,
        };

        let router = EguiRouter::builder()
            .route("/", home_route)
            .route("/about", about_route)
            .route("/help", help_route)
            .default_path("/")
            .transition(TransitionConfig::fade().with_duration(0.01))
            .build(&mut state);
        state.router = Some(router);
        state
    }

    fn poll_channels(&mut self, ctx: &egui::Context) {
        // Update system volume from shared atomic written by the background thread
        self.system_volume = f32::from_bits(self.shared_volume.load(Ordering::Relaxed));

        // Update LSNR
        if !self.r_lsnr.is_empty() {
            let mut lsnr = 0.0f32;
            let mut n = 0usize;
            while let Ok(v) = self.r_lsnr.try_recv() {
                lsnr += v;
                n += 1;
            }
            if n > 0 {
                self.lsnr = lsnr / n as f32;
            }
        }

        // Update noisy spectrogram
        if !self.r_noisy.is_empty() {
            let specs: Vec<_> = self.r_noisy.try_iter().collect();
            let n = specs.len();
            if let Some(spec) = self.spec_noisy.as_mut() {
                spec.update(specs.into_iter(), n);
                self.noisy_texture = Some(ctx.load_texture(
                    "noisy_spec",
                    spec.to_color_image(self.dark_mode),
                    egui::TextureOptions::LINEAR,
                ));
            }
        }

        // Update enhanced spectrogram
        if !self.r_enh.is_empty() {
            let specs: Vec<_> = self.r_enh.try_iter().collect();
            let n = specs.len();
            if let Some(spec) = self.spec_enh.as_mut() {
                spec.update(specs.into_iter(), n);
                self.enh_texture = Some(ctx.load_texture(
                    "enh_spec",
                    spec.to_color_image(self.dark_mode),
                    egui::TextureOptions::LINEAR,
                ));
            }
        }

        // Check device events
        if let Some(ref r_device_event) = self.r_device_event {
            if let Ok(event) = r_device_event.try_recv() {
                self.handle_device_lost(event);
            }
        }
    }

    fn handle_device_lost(&mut self, event: DeviceEvent) {
        log::warn!("Audio device lost during capture, resetting UI and refreshing devices.");
        self.df_worker = None;
        self.spec_noisy = None;
        self.spec_enh = None;
        self.noisy_texture = None;
        self.enh_texture = None;
        self.refresh_devices();
        match event {
            DeviceEvent::InputLost => {
                self.input_device = match find_cable() {
                    Ok(name) => {
                        log::info!("Auto-selected input device: {name}");
                        Some(name)
                    }
                    Err(e) => {
                        log::warn!("Could not auto-select input device: {e}");
                        None
                    }
                };
            }
            DeviceEvent::OutputLost => {
                self.output_device = match set_output_device() {
                    Ok(name) => {
                        log::info!("Auto-selected output device: {name}");
                        Some(name)
                    }
                    Err(e) => {
                        log::warn!("Could not auto-select output device: {e}");
                        None
                    }
                };
            }
        }
    }

    fn start_capture(&mut self, input_device: Option<String>, output_device: Option<String>) {
        let (s_lsnr, r_lsnr) = unbounded();
        let (s_noisy, r_noisy) = unbounded();
        let (s_enh, r_enh) = unbounded();
        let (s_controls, r_controls) = unbounded();
        let (s_device_event, r_device_event) = unbounded();

        log::info!("Using model: {}", self.selected_model.label());
        let result = match self.selected_model {
            ModelChoice::DeepFilter(kind) => DeepFilterCapture::new(
                kind,
                input_device,
                output_device,
                Some(s_lsnr),
                Some(s_noisy),
                Some(s_enh),
                Some(r_controls),
                Some(s_device_event),
                self.shared_volume.clone(),
                self.smoother_enabled.clone(),
            )
            .map(|w| (w.sr, w.frame_size, w.freq_size, CaptureWorker::DeepFilter(w))),
            ModelChoice::DpdfNet(kind) => DpdfNetCapture::new(
                kind,
                input_device,
                output_device,
                Some(s_noisy),
                Some(s_enh),
                Some(r_controls),
                Some(s_device_event),
                self.shared_volume.clone(),
                self.smoother_enabled.clone(),
            )
            .map(|w| (w.sr, w.frame_size, w.freq_size, CaptureWorker::DpdfNet(w))),
        };
        match result {
            Ok((sr, frame_size, freq_size, df_worker)) => {
                let w = (sr / frame_size * 10) as u32;
                let freq_res = sr / 2 / (freq_size - 1);
                let h = (8000 / freq_res) as u32;

                self.spec_noisy = Some(SpecImage::new(w, h, -100., -10.));
                self.spec_enh = Some(SpecImage::new(w, h, -100., -10.));
                self.df_worker = Some(df_worker);
                self.lsnr = 0.;
                self.r_lsnr = r_lsnr;
                self.r_noisy = r_noisy;
                self.r_enh = r_enh;
                self.s_controls = s_controls;
                self.r_device_event = Some(r_device_event);
                // Push current slider values so a fresh start actually honors them:
                // init_df builds DeepFilterNet with post-filtering off, and DPDFNet
                // starts without an attenuation limit.
                if self.selected_model.is_deep_filter() {
                    self.s_controls
                        .send((DfControl::PostFilterBeta, self.post_filter_beta))
                        .ok();
                } else {
                    self.s_controls
                        .send((DfControl::AttenLim, self.atten_lim))
                        .ok();
                }
            }
            Err(e) => {
                log::error!("Failed to initialize audio capturing: {:#}", e);
                self.df_worker = None;
                self.refresh_devices();
                if let Some(dev_err) = e.downcast_ref::<capture::DeviceSelectError>() {
                    match dev_err {
                        capture::DeviceSelectError::InputNotFound(_)
                        | capture::DeviceSelectError::NoInputProvided => {
                            log::debug!("Input device not found/provided");
                            self.auto_select_input_device();
                        }
                        capture::DeviceSelectError::OutputNotFound(_)
                        | capture::DeviceSelectError::NoOutputProvided => {
                            log::debug!("Output device not found/provided");
                            self.auto_select_output_device();
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn stop_capture(&mut self) {
        if let Some(worker) = self.df_worker.as_mut() {
            log::info!("Stopping cleaning ...");
            worker.should_stop().expect("Failed to stop DF worker");
        }
        self.df_worker = None;
    }

    fn refresh_devices(&mut self) {
        log::info!("Refreshing devices...");
        self.available_input_devices = get_input_devices().unwrap_or_else(|e| {
            log::error!("Failed to get input devices: {}", e);
            vec![]
        });
        self.available_output_devices = get_output_devices().unwrap_or_else(|e| {
            log::error!("Failed to get output devices: {}", e);
            vec![]
        });
    }

    fn auto_select_input_device(&mut self) {
        self.input_device = match find_cable() {
            Ok(name) => {
                log::info!("Auto-selected input device: {name}");
                Some(name)
            }
            Err(e) => {
                log::warn!("Could not auto-select input device: {e}");
                None
            }
        };
    }

    fn auto_select_output_device(&mut self) {
        self.output_device = match set_output_device() {
            Ok(name) => {
                log::info!("Auto-selected output device: {name}");
                Some(name)
            }
            Err(e) => {
                log::warn!("Could not auto-select output device: {e}");
                None
            }
        };
    }
}

fn show_snr_gauge(ui: &mut egui::Ui, lsnr: f32) {
    const SNR_MIN: f32 = -20.0;
    const SNR_MAX: f32 = 20.0;
    let fill_frac = ((lsnr - SNR_MIN) / (SNR_MAX - SNR_MIN)).clamp(0.0, 1.0);
    let gauge_w = 28.0;
    let gauge_h = 100.0;

    let bg_color = ui.visuals().extreme_bg_color;
    let border_stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    let tick_color = ui.visuals().text_color();

    ui.vertical(|ui| {
        ui.label(egui::RichText::new("SNR").small());
        let (rect, _) = ui.allocate_exact_size(egui::vec2(gauge_w, gauge_h), egui::Sense::hover());
        let painter = ui.painter();

        // Background
        painter.rect_filled(rect, 3.0, bg_color);

        // Fill bar (grows upward)
        if fill_frac > 0.0 {
            let fill_h = (gauge_h - 2.0) * fill_frac;
            let fill_rect = egui::Rect::from_min_max(
                egui::pos2(rect.min.x + 2.0, rect.max.y - 1.0 - fill_h),
                egui::pos2(rect.max.x - 2.0, rect.max.y - 1.0),
            );
            painter.rect_filled(fill_rect, 2.0, snr_bar_color(fill_frac));
        }

        // Threshold tick at 5 dB (music-detected boundary)
        let thresh_frac = (5.0 - SNR_MIN) / (SNR_MAX - SNR_MIN);
        let tick_y = rect.max.y - gauge_h * thresh_frac;
        painter.line_segment(
            [
                egui::pos2(rect.min.x, tick_y),
                egui::pos2(rect.max.x, tick_y),
            ],
            egui::Stroke::new(1.0, tick_color),
        );

        // Border
        painter.rect_stroke(rect, 3.0, border_stroke, egui::StrokeKind::Middle);

        ui.add_sized(
            egui::vec2(56.0, 14.0),
            egui::Label::new(egui::RichText::new(format!("{:+.1} dB", lsnr)).small()),
        );
    });
}

fn snr_bar_color(t: f32) -> egui::Color32 {
    // red → orange → green as SNR increases
    let (r, g) = if t < 0.5 {
        let s = t * 2.0;
        (220u8, (60.0 + s * 160.0) as u8)
    } else {
        let s = (t - 0.5) * 2.0;
        ((220.0 - s * 140.0) as u8, 220u8)
    };
    egui::Color32::from_rgb(r, g, 20)
}

fn show_volume_knob(ui: &mut egui::Ui, volume: f32) {
    let size = 80.0;

    let knob_bg = ui.visuals().widgets.noninteractive.bg_fill;
    let knob_stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    let track_bg = ui.visuals().widgets.inactive.bg_fill;
    let needle_color = ui.visuals().text_color();
    let dot_color = ui.visuals().widgets.noninteractive.fg_stroke.color;

    ui.vertical(|ui| {
        ui.label(egui::RichText::new("Volume").small());
        let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
        let painter = ui.painter();
        let center = rect.center();
        let radius = size / 2.0 - 5.0;

        // Background circle
        painter.circle_filled(center, radius, knob_bg);
        painter.circle_stroke(center, radius, knob_stroke);

        // Knob arc: 225° start, 270° total sweep (clockwise)
        let start_angle = std::f32::consts::PI * 1.25;
        let total_sweep = std::f32::consts::PI * 1.5;
        let track_r = radius * 0.68;
        let n = 48usize;

        // Background track
        let bg_points: Vec<egui::Pos2> = (0..=n)
            .map(|i| {
                let a = start_angle + (i as f32 / n as f32) * total_sweep;
                egui::pos2(center.x + track_r * a.cos(), center.y + track_r * a.sin())
            })
            .collect();
        painter.add(egui::Shape::line(
            bg_points,
            egui::Stroke::new(4.0, track_bg),
        ));

        // Filled arc
        if volume > 0.001 {
            let filled_segs = ((n as f32 * volume) as usize + 1).min(n);
            let val_points: Vec<egui::Pos2> = (0..=filled_segs)
                .map(|i| {
                    let a = start_angle + (i as f32 / n as f32) * total_sweep;
                    egui::pos2(center.x + track_r * a.cos(), center.y + track_r * a.sin())
                })
                .collect();
            painter.add(egui::Shape::line(
                val_points,
                egui::Stroke::new(4.0, egui::Color32::from_rgb(90, 160, 230)),
            ));
        }

        // Indicator needle
        let needle_angle = start_angle + volume * total_sweep;
        let p_inner = egui::pos2(
            center.x + radius * 0.28 * needle_angle.cos(),
            center.y + radius * 0.28 * needle_angle.sin(),
        );
        let p_outer = egui::pos2(
            center.x + radius * 0.80 * needle_angle.cos(),
            center.y + radius * 0.80 * needle_angle.sin(),
        );
        painter.line_segment([p_inner, p_outer], egui::Stroke::new(2.0, needle_color));
        painter.circle_filled(center, 3.5, dot_color);

        // Value label
        let label = if volume == 0.0 {
            "Muted".to_string()
        } else {
            format!("{}%", (volume * 100.0).round() as u32)
        };
        ui.label(egui::RichText::new(label).small());
    });
}

impl eframe::App for LiveMusicRemover {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_channels(ctx);
        ctx.request_repaint_after(Duration::from_millis(20));
        ctx.set_visuals(if self.dark_mode {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        });

        if self.dark_mode != self.last_dark_mode {
            self.last_dark_mode = self.dark_mode;
            if let (Some(spec), Some(tex_slot)) =
                (self.spec_noisy.as_ref(), self.noisy_texture.as_mut())
            {
                *tex_slot = ctx.load_texture(
                    "noisy_spec",
                    spec.to_color_image(self.dark_mode),
                    egui::TextureOptions::LINEAR,
                );
            }
            if let (Some(spec), Some(tex_slot)) =
                (self.spec_enh.as_ref(), self.enh_texture.as_mut())
            {
                *tex_slot = ctx.load_texture(
                    "enh_spec",
                    spec.to_color_image(self.dark_mode),
                    egui::TextureOptions::LINEAR,
                );
            }
        }

        if self.logo_texture.is_none() {
            let bytes = include_bytes!("../assets/logo.svg");
            if let Ok(color_image) = egui_extras::image::load_svg_bytes(bytes) {
                self.logo_texture =
                    Some(ctx.load_texture("logo", color_image, egui::TextureOptions::LINEAR));
            }
        }

        let is_running = self.df_worker.is_some();
        let start_enabled =
            !is_running && self.input_device.is_some() && self.output_device.is_some();

        // ── Router: take out for this frame ──────────────────────────────────
        let mut router = self.router.take().unwrap();

        // ── Top navigation bar ────────────────────────────────────────────────
        let mut navigate_to: Option<String> = None;
        egui::TopBottomPanel::top("nav_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // ui.add_space(4.0);
                if ui
                    .selectable_label(self.current_page == Page::Main, "🔨 Main")
                    .clicked()
                {
                    self.current_page = Page::Main;
                    navigate_to = Some("/".to_string());
                }
                if ui
                    .selectable_label(self.current_page == Page::Help, "❓  Help")
                    .clicked()
                {
                    self.current_page = Page::Help;
                    navigate_to = Some("/help".to_string());
                }
                if ui
                    .selectable_label(self.current_page == Page::About, "ℹ  About")
                    .clicked()
                {
                    self.current_page = Page::About;
                    navigate_to = Some("/about".to_string());
                }
            });
        });
        if let Some(route) = navigate_to {
            router.navigate(self, route).ok();
        }

        // ── Left panel: all controls (main page only) ─────────────────────────
        if self.current_page == Page::Main {
            egui::SidePanel::left("controls_panel")
            .resizable(false)
            .exact_width(420.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(8.0);

                    // Title row with logo
                    ui.horizontal(|ui| {
                        let logo_size = 42.0;

                        let (response, painter) = ui.allocate_painter(
                            egui::Vec2::splat(logo_size),
                            egui::Sense::hover(),
                        );
                        let logo_rect = egui::Rect::from_center_size(
                            response.rect.center(),
                            egui::Vec2::splat(logo_size),
                        );

                        let tint = if is_running {
                            egui::Color32::WHITE // original color
                        } else {
                            egui::Color32::from_rgba_unmultiplied(180, 180, 180, 120) // dimmed
                        };

                        if let Some(tex) = &self.logo_texture {
                            painter.image(
                                tex.id(),
                                logo_rect,
                                egui::Rect::from_min_max(
                                    egui::pos2(0.0, 0.0),
                                    egui::pos2(1.0, 1.0),
                                ),
                                tint,
                            );
                        }


                        // ui.heading(egui::RichText::new("Live Music Remover").strong());
                        ui.vertical(|ui| {
                            ui.label(
                                egui::RichText::new("Live Music Remover")
                                    .font(egui::FontId::proportional(30.0))
                                    .strong(),
                            );
                            ui.label(
                                egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                                    .small()
                                    .color(egui::Color32::GRAY),
                            );
                            if let Ok(info) = self.update_info.lock() {
                                if let Some((latest, url)) = info.as_ref() {
                                    ui.add(
                                        egui::Hyperlink::from_label_and_url(
                                            egui::RichText::new(format!("New version v{} is available", latest.trim_start_matches('v')))
                                                .small()
                                                .color(egui::Color32::YELLOW),
                                            url,
                                        ),
                                    );
                                }
                            }
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("Exit").clicked() {
                                if let Some(worker) = self.df_worker.as_mut() {
                                    worker.should_stop().ok();
                                }
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                            let theme_icon = if self.dark_mode { "☀" } else { "🌙" };
                            if ui.button(theme_icon).on_hover_text("Toggle light/dark mode").clicked() {
                                self.dark_mode = !self.dark_mode;
                            }
                        });
                    });

                    ui.add_space(6.0);

                    // Devices card
                    ui.group(|ui| {
                        ui.label(egui::RichText::new("Audio Devices").strong());
                        ui.add_space(4.0);

                        let selected_input = self.input_device.clone().unwrap_or_default();
                        let input_devices = self.available_input_devices.clone();
                        let current_input = self.input_device.clone();
                        ui.label("Input device:");
                        egui::ComboBox::from_id_salt("input_device")
                            .width(380.0)
                            .selected_text(&selected_input)
                            .show_ui(ui, |ui| {
                                for device in &input_devices {
                                    let is_selected =
                                        current_input.as_deref() == Some(device.as_str());
                                    if ui.selectable_label(is_selected, device).clicked() {
                                        let device = device.clone();
                                        self.refresh_devices();
                                        if self.df_worker.is_some() {
                                            self.stop_capture();
                                            self.spec_noisy = None;
                                            self.spec_enh = None;
                                            self.noisy_texture = None;
                                            self.enh_texture = None;
                                            if device_exists(
                                                Some(device.as_str()),
                                                &self.available_input_devices,
                                            ) {
                                                self.input_device = Some(device);
                                            } else {
                                                self.auto_select_input_device();
                                            }
                                            let inp = self.input_device.clone();
                                            let out = self.output_device.clone();
                                            self.start_capture(inp, out);
                                        } else if device_exists(
                                            Some(device.as_str()),
                                            &self.available_input_devices,
                                        ) {
                                            self.input_device = Some(device);
                                        } else {
                                            self.auto_select_input_device();
                                        }
                                    }
                                }
                            });

                        ui.add_space(4.0);

                        let selected_output = self.output_device.clone().unwrap_or_default();
                        let output_devices = self.available_output_devices.clone();
                        let current_output = self.output_device.clone();
                        ui.label("Output device:");
                        egui::ComboBox::from_id_salt("output_device")
                            .width(380.0)
                            .selected_text(&selected_output)
                            .show_ui(ui, |ui| {
                                for device in &output_devices {
                                    let is_selected =
                                        current_output.as_deref() == Some(device.as_str());
                                    if ui.selectable_label(is_selected, device).clicked() {
                                        let device = device.clone();
                                        self.refresh_devices();
                                        if self.df_worker.is_some() {
                                            self.stop_capture();
                                            self.spec_noisy = None;
                                            self.spec_enh = None;
                                            self.noisy_texture = None;
                                            self.enh_texture = None;
                                            if device_exists(
                                                Some(device.as_str()),
                                                &self.available_output_devices,
                                            ) {
                                                self.output_device = Some(device);
                                            } else {
                                                self.auto_select_output_device();
                                            }
                                            let inp = self.input_device.clone();
                                            let out = self.output_device.clone();
                                            self.start_capture(inp, out);
                                        } else if device_exists(
                                            Some(device.as_str()),
                                            &self.available_output_devices,
                                        ) {
                                            self.output_device = Some(device);
                                        } else {
                                            self.auto_select_output_device();
                                        }
                                    }
                                }
                            });

                        ui.add_space(4.0);
                        if ui.button("Refresh Devices").clicked() {
                            self.refresh_devices();
                            if !device_exists(
                                self.input_device.as_deref(),
                                &self.available_input_devices,
                            ) {
                                self.auto_select_input_device();
                            }
                            if !device_exists(
                                self.output_device.as_deref(),
                                &self.available_output_devices,
                            ) {
                                self.auto_select_output_device();
                            }
                        }
                    });

                    ui.add_space(8.0);

                    // Model selection (disabled while capturing)
                    ui.group(|ui| {
                        ui.label(egui::RichText::new("Model").strong());
                        ui.add_space(4.0);
                        ui.add_enabled_ui(!is_running, |ui| {
                            egui::ComboBox::from_id_salt("model_select")
                                .width(380.0)
                                .selected_text(self.selected_model.label())
                                .show_ui(ui, |ui| {
                                    for choice in ModelChoice::ALL {
                                        ui.selectable_value(
                                            &mut self.selected_model,
                                            choice,
                                            choice.label(),
                                        );
                                    }
                                });
                        });
                        // if is_running {
                        //     ui.colored_label(
                        //         egui::Color32::from_rgb(160, 160, 160),
                        //         "Stop capturing to change model.",
                        //     );
                        // }
                    });

                    ui.add_space(8.0);

                    // Start / Stop buttons
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                start_enabled,
                                egui::Button::new("▶  Start").min_size(egui::vec2(100.0, 32.0)),
                            )
                            .clicked()
                        {
                            let inp = self.input_device.clone();
                            let out = self.output_device.clone();
                            self.start_capture(inp, out);
                        }
                        if ui
                            .add_enabled(
                                is_running,
                                egui::Button::new("■  Stop").min_size(egui::vec2(100.0, 32.0)),
                            )
                            .clicked()
                        {
                            self.stop_capture();
                            self.spec_noisy = None;
                            self.spec_enh = None;
                            self.noisy_texture = None;
                            self.enh_texture = None;
                        }
                        if !is_running && !start_enabled {
                            ui.colored_label(
                                egui::Color32::from_rgb(255, 80, 80),
                                "Select devices to start.",
                            );
                        }
                    });

                    ui.add_space(8.0);
                    ui.separator();

                    ui.horizontal(|ui| {
                        // Left column: sliders, constrained so the right column always fits
                        let right_col_w = 100.0;
                        let left_col_w = ui.available_width()
                            - right_col_w
                            - ui.spacing().item_spacing.x * 2.0;
                        ui.vertical(|ui| {
                            ui.set_max_width(left_col_w);

                            // Controls card
                            ui.group(|ui| {
                                ui.label(egui::RichText::new("Controls").strong());
                                ui.add_space(4.0);

                                ui.label("Noise Attenuation [dB]");
                                if ui
                                    .add(egui::Slider::new(&mut self.atten_lim, 0.0..=100.0))
                                    .on_hover_text(
                                        "Controls how aggressively music is removed. \
                                         Higher values remove more music but may affect voice quality.",
                                    )
                                    .changed()
                                {
                                    self.s_controls
                                        .send((DfControl::AttenLim, self.atten_lim))
                                        .ok();
                                }

                                // DPDFNet has no post filter.
                                if self.selected_model.is_deep_filter() {
                                    ui.add_space(4.0);
                                    ui.label("Post Filter Beta");
                                    if ui
                                        .add(
                                            egui::Slider::new(&mut self.post_filter_beta, 0.0..=1.0)
                                                .step_by(0.001),
                                        )
                                        .on_hover_text(
                                            "Smooths the filter's effect on the audio. \
                                             Increase if you hear musical artefacts in the output.",
                                        )
                                        .changed()
                                    {
                                        self.s_controls
                                            .send((DfControl::PostFilterBeta, self.post_filter_beta))
                                            .ok();
                                    }

                                }

                                ui.add_space(4.0);
                                let mut smoother_on =
                                    self.smoother_enabled.load(Ordering::Relaxed);
                                if ui
                                    .checkbox(&mut smoother_on, "Transient Smoother")
                                    .on_hover_text(
                                        "Ducks short loud spikes (musical-noise blips) relative \
                                         to the recent level. Toggle off to A/B against the raw \
                                         model output.",
                                    )
                                    .changed()
                                {
                                    self.smoother_enabled
                                        .store(smoother_on, Ordering::Relaxed);
                                }

                                // ui.label("Threshold Min [dB]");
                                //     if ui
                                //         .add(
                                //             egui::Slider::new(
                                //                 &mut self.min_threshdb,
                                //                 -15.0..=35.0,
                                //             )
                                //             .step_by(1.0),
                                //         )
                                //         .changed()
                                //     {
                                //         self.s_controls
                                //             .send((DfControl::MinThreshDb, self.min_threshdb))
                                //             .ok();
                                //     }
                                //     ui.label("Threshold ERB Max [dB]");
                                //     if ui
                                //         .add(
                                //             egui::Slider::new(
                                //                 &mut self.max_erbthreshdb,
                                //                 -15.0..=35.0,
                                //             )
                                //             .step_by(1.0),
                                //         )
                                //         .changed()
                                //     {
                                //         self.s_controls
                                //             .send((
                                //                 DfControl::MaxErbThreshDb,
                                //                 self.max_erbthreshdb,
                                //             ))
                                //             .ok();
                                //     }
                                //     ui.label("Threshold DF Max [dB]");
                                //     if ui
                                //         .add(
                                //             egui::Slider::new(
                                //                 &mut self.max_dfthreshdb,
                                //                 -15.0..=35.0,
                                //             )
                                //             .step_by(1.0),
                                //         )
                                //         .changed()
                                //     {
                                //         self.s_controls
                                //             .send((DfControl::MaxDfThreshDb, self.max_dfthreshdb))
                                //             .ok();
                                //     }
                            });

                            // Visualization Controls card
                            ui.group(|ui| {
                                ui.label(egui::RichText::new("Visualisation controls").strong());
                                ui.add_space(4.0);

                                ui.label("Frequency Axis Scale");
                                if ui.add(egui::Slider::new(&mut self.freq_axis_scale, 0.5..=4.0).step_by(0.01)).changed() {
                                    // No action needed, just triggers repaint
                                }

                            });

                            // // Status (below controls, in the same left column)
                            // if is_running {
                            //     ui.add_space(4.0);
                            //     ui.horizontal(|ui| {
                            //         ui.spinner();
                            //         let status = if self.lsnr < 5.0 {
                            //             "Music detected — removing"
                            //         } else {
                            //             "No music detected"
                            //         };
                            //         ui.label(status);
                            //     });
                            // }
                        });

                        // Right column: SNR gauge + Volume knob
                        ui.group(|ui| {
                            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                show_volume_knob(ui, self.system_volume);
                                // DPDFNet has no SNR estimate.
                                if self.selected_model.is_deep_filter() {
                                    ui.add_space(8.0);
                                    if is_running {
                                        show_snr_gauge(ui, self.lsnr);
                                    } else {
                                        show_snr_gauge(ui, 0.0);
                                    }
                                }
                            });
                        });
                    });
                });
            });
        } // end if Page::Main

        // ── Central panel: router-controlled content ──────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            router.ui(ui, self);
        });

        self.router = Some(router);
    }
}

// ── Route handlers ────────────────────────────────────────────────────────────

fn home_route() -> impl Route<LiveMusicRemover> {
    |ui: &mut egui::Ui, state: &mut LiveMusicRemover| {
        let is_running = state.df_worker.is_some();
        egui::CollapsingHeader::new("Visualisation")
            .default_open(true)
            .show(ui, |ui| {
                if !is_running {
                    ui.add_space(20.0);
                    ui.vertical_centered(|ui| {
                        ui.label("Start processing to see the audio visualisation.");
                    });
                    return;
                }
                let base_spec_h = 1000.0;
                let spec_h = base_spec_h * state.freq_axis_scale;
                let spec_w = 2000.0;

                ui.vertical(|ui| {
                    if let Some(ref texture) = state.noisy_texture {
                        ui.vertical(|ui| {
                            ui.label("Before");
                            ui.add(
                                egui::Image::new(texture)
                                    .fit_to_exact_size(egui::vec2(spec_h, spec_w)),
                            );
                        });
                    }

                    ui.add_space(4.0);

                    if let Some(ref texture) = state.enh_texture {
                        ui.vertical(|ui| {
                            ui.label("After");
                            ui.add(
                                egui::Image::new(texture)
                                    .fit_to_exact_size(egui::vec2(spec_h, spec_w)),
                            );
                        });
                    }
                });
            });
    }
}

fn about_route() -> impl Route<LiveMusicRemover> {
    |ui: &mut egui::Ui, state: &mut LiveMusicRemover| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(16.0);
            ui.heading("About Live Music Remover");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                if let Ok(info) = state.update_info.lock() {
                    if let Some((latest, url)) = info.as_ref() {
                        ui.add(
                            egui::Hyperlink::from_label_and_url(
                                egui::RichText::new(format!("New version v{} is available", latest.trim_start_matches('v')))
                                    .color(egui::Color32::YELLOW),
                                url,
                            ),
                        );
                    }
                }
            });
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(
                "Live Music Remover is a real-time audio processing tool that removes \
                 background music and noise from a live audio stream while preserving speech. \
                 It uses DeepFilterNet, a deep neural network trained for noise and \
                 music suppression.",
            );
            ui.add_space(12.0);
            ui.label(
                "Audio is captured from a virtual audio cable input, processed \
                 frame-by-frame through the DeepFilterNet model, and played back \
                 on your chosen output device.",
            );
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Built with").strong());
            ui.add_space(4.0);
            ui.label("• DeepFilterNet  — neural network for music/noise suppression");
            ui.label("• DPDFNet        — speech enhancement model by Ceva (Apache-2.0)");
            ui.label("• ONNX Runtime   — inference engine for DPDFNet");
            ui.label("• egui / eframe  — immediate-mode GUI framework");
            ui.label("• cpal           — cross-platform audio I/O");
            ui.label("• rubato         — high-quality audio resampling");
            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Source code").strong());
            ui.add_space(4.0);
            ui.label("https://github.com/ilyasben26/live-music-remover");
        });
    }
}

fn help_route() -> impl Route<LiveMusicRemover> {
    |ui: &mut egui::Ui, _state: &mut LiveMusicRemover| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(16.0);
            ui.label(
                egui::RichText::new("How to use?")
                    .font(egui::FontId::proportional(30.0))
                    .strong(),
            );
            ui.add_space(8.0);
            ui.separator();

            ui.label("TODO: put link to youtube video tutorial");

            ui.add_space(12.0);
            ui.label(egui::RichText::new("Step 1 — Install a virtual audio cable").strong());
            ui.add_space(4.0);
            ui.label(
                "Install VB-CABLE (https://vb-audio.com/Cable). This creates a virtual \
                 input/output device pair that lets you route audio into Live Music Remover.",
            );

            ui.add_space(12.0);
            ui.label(egui::RichText::new("Step 2 — Route your audio").strong());
            ui.add_space(4.0);
            ui.label(
                "In your streaming software or system settings, set the playback device \
                 to \"CABLE Input\". This sends the audio to Live Music Remover for processing.",
            );

            ui.add_space(12.0);
            ui.label(egui::RichText::new("Step 3 — Select devices").strong());
            ui.add_space(4.0);
            ui.label(
                "• Input device:  select \"CABLE Output\" (what your source is playing into).",
            );
            ui.label("• Output device: select your headphones or speakers.");
            ui.label("  Use \"Refresh Devices\" if a device is missing from the list.");

            ui.add_space(12.0);
            ui.label(egui::RichText::new("Step 4 — Choose a model").strong());
            ui.add_space(4.0);
            ui.label("• Standard");
            ui.label("• Low Latency - Recommended.");
            ui.label("• DPDFNet-2 48 kHz HR - alternative full-band model.");
            ui.label(
                "• DPDFNet-8 48 kHz HR - larger DPDFNet, separates voice from music better.                  Uses the most CPU (two cores).",
            );
            ui.label(
                "It's a good idea to try both and see which one removes music best in your setup.",
            );

            ui.add_space(12.0);
            ui.label(egui::RichText::new("Step 5 — Start processing").strong());
            ui.add_space(4.0);
            ui.label("Click ▶ Start. Processed audio plays on your output device.");

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Controls").strong());
            ui.add_space(4.0);
            ui.label(
                "• Noise Attenuation [dB] — how aggressively music is removed. 0 means no filtering, 100 means maximum filtering. It's recommended to keep it at 100.",
            );
            ui.label(
                "• Post Filter Beta — smooths the filter output. \
                 Increase if you hear musical artefacts.",
            );
            ui.label("• Frequency Axis Scale — zooms the spectrogram visualisation.");

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("SNR Gauge").strong());
            ui.add_space(4.0);
            ui.label(
                "TODO: Add SNR gauge description",
            );

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Reporting bugs / issues").strong());
            ui.add_space(4.0);
            ui.label(
                "TODO: Add instructions and motivations for submiting a github issue",
            )
            
        });
    }
}
