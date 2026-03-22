#![cfg_attr(
    all(target_os = "windows", not(feature = "dev-console")),
    windows_subsystem = "windows"
)]

use std::env;
use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, ValueHint};
use crossbeam_channel::{unbounded, Receiver};
use eframe::egui;
use image_rs::{imageops, Rgba, RgbaImage};

mod capture;
mod cmap;
mod devices;
use capture::*;

use crate::devices::{
    device_exists, find_cable, get_input_devices, get_output_devices, set_output_device,
};

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model tar.gz
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,
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
    if args.model.is_some() {
        unsafe { MODEL_PATH = args.model }
    }

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
            .with_title("DeepFilterNet Demo")
            .with_inner_size([1100.0, 900.0]),
        ..Default::default()
    };

    eframe::run_native(
        "DeepFilterNet Demo",
        options,
        Box::new(|_cc| Ok(Box::new(LiveMusicRemover::new()))),
    )
}

struct LiveMusicRemover {
    df_worker: Option<DeepFilterCapture>,
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
    r_lsnr: RecvLsnr,
    r_noisy: RecvSpec,
    r_enh: RecvSpec,
    s_controls: SendControl,
    input_device: Option<String>,
    output_device: Option<String>,
    available_input_devices: Vec<String>,
    available_output_devices: Vec<String>,
    r_device_event: Option<Receiver<DeviceEvent>>,
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
    fn to_color_image(&self) -> egui::ColorImage {
        let rotated = imageops::rotate270(&self.im);
        let pixels: Vec<egui::Color32> = rotated
            .pixels()
            .map(|p| egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
            .collect();
        egui::ColorImage {
            size: [self.n_frames as usize, self.n_freqs as usize],
            pixels,
        }
    }
}

impl LiveMusicRemover {
    fn new() -> Self {
        let (_s_lsnr, r_lsnr) = unbounded();
        let (_s_noisy, r_noisy) = unbounded();
        let (_s_enh, r_enh) = unbounded();
        let (s_controls, _r_controls) = unbounded();
        let (_s_device_event, r_device_event) = unbounded();

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

        Self {
            df_worker: None,
            lsnr: 0.,
            atten_lim: 100.,
            post_filter_beta: 0.,
            min_threshdb: -15.,
            max_erbthreshdb: 35.,
            max_dfthreshdb: 35.,
            spec_noisy: None,
            spec_enh: None,
            noisy_texture: None,
            enh_texture: None,
            r_lsnr,
            r_noisy,
            r_enh,
            s_controls,
            input_device,
            output_device,
            available_input_devices,
            available_output_devices,
            r_device_event: Some(r_device_event),
        }
    }

    fn poll_channels(&mut self, ctx: &egui::Context) {
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
                    spec.to_color_image(),
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
                    spec.to_color_image(),
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

        let model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
        log::info!("Using model path: {:?}", model_path);
        match DeepFilterCapture::new(
            model_path,
            input_device,
            output_device,
            Some(s_lsnr),
            Some(s_noisy),
            Some(s_enh),
            Some(r_controls),
            Some(s_device_event),
        ) {
            Ok(df_worker) => {
                let w = (df_worker.sr / df_worker.frame_size * 10) as u32;
                let freq_res = df_worker.sr / 2 / (df_worker.freq_size - 1);
                let h = (8000 / freq_res) as u32;

                self.spec_noisy = Some(SpecImage::new(w, h, -100., -10.));
                self.spec_enh = Some(SpecImage::new(w, h, -100., -10.));
                self.df_worker = Some(df_worker);
                self.r_lsnr = r_lsnr;
                self.r_noisy = r_noisy;
                self.r_enh = r_enh;
                self.s_controls = s_controls;
                self.r_device_event = Some(r_device_event);
            }
            Err(e) => {
                log::error!("Failed to initialize DeepFilterNet audio capturing: {}", e);
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

impl eframe::App for LiveMusicRemover {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll channels for new data
        self.poll_channels(ctx);

        // Request repaint at 20ms intervals for real-time updates
        ctx.request_repaint_after(Duration::from_millis(20));

        egui::CentralPanel::default().show(ctx, |ui| {
            // Title bar row
            ui.horizontal(|ui| {
                ui.heading("Live Music Remover");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Exit").clicked() {
                        if let Some(worker) = self.df_worker.as_mut() {
                            worker.should_stop().ok();
                        }
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });

            ui.separator();

            // Device selection
            let selected_input = self.input_device.clone().unwrap_or_default();
            let input_devices = self.available_input_devices.clone();
            let current_input = self.input_device.clone();
            egui::ComboBox::from_label("Input device")
                .selected_text(&selected_input)
                .show_ui(ui, |ui| {
                    for device in &input_devices {
                        let is_selected = current_input.as_deref() == Some(device.as_str());
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

            let selected_output = self.output_device.clone().unwrap_or_default();
            let output_devices = self.available_output_devices.clone();
            let current_output = self.output_device.clone();
            egui::ComboBox::from_label("Output device")
                .selected_text(&selected_output)
                .show_ui(ui, |ui| {
                    for device in &output_devices {
                        let is_selected = current_output.as_deref() == Some(device.as_str());
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

            if ui.button("Refresh Devices").clicked() {
                self.refresh_devices();
                if !device_exists(self.input_device.as_deref(), &self.available_input_devices) {
                    self.auto_select_input_device();
                }
                if !device_exists(
                    self.output_device.as_deref(),
                    &self.available_output_devices,
                ) {
                    self.auto_select_output_device();
                }
            }

            ui.separator();

            // Start / Stop buttons
            let start_enabled = self.df_worker.is_none()
                && self.input_device.is_some()
                && self.output_device.is_some();
            let stop_enabled = self.df_worker.is_some();

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(start_enabled, egui::Button::new("Start"))
                    .clicked()
                {
                    let inp = self.input_device.clone();
                    let out = self.output_device.clone();
                    self.start_capture(inp, out);
                }
                if ui
                    .add_enabled(stop_enabled, egui::Button::new("Stop"))
                    .clicked()
                {
                    self.stop_capture();
                    self.spec_noisy = None;
                    self.spec_enh = None;
                    self.noisy_texture = None;
                    self.enh_texture = None;
                }
                if self.df_worker.is_none() && !start_enabled {
                    ui.colored_label(
                        egui::Color32::from_rgb(255, 60, 60),
                        "Select input and output devices to start.",
                    );
                }
            });

            ui.separator();

            // Sliders
            #[cfg(feature = "thresholds")]
            {
                ui.label("Threshold Min [dB]");
                if ui
                    .add(egui::Slider::new(&mut self.min_threshdb, -15.0..=35.0).step_by(1.0))
                    .changed()
                {
                    self.s_controls
                        .send((DfControl::MinThreshDb, self.min_threshdb))
                        .ok();
                }

                ui.label("Threshold ERB Max [dB]");
                if ui
                    .add(egui::Slider::new(&mut self.max_erbthreshdb, -15.0..=35.0).step_by(1.0))
                    .changed()
                {
                    self.s_controls
                        .send((DfControl::MaxErbThreshDb, self.max_erbthreshdb))
                        .ok();
                }

                ui.label("Threshold DF Max [dB]");
                if ui
                    .add(egui::Slider::new(&mut self.max_dfthreshdb, -15.0..=35.0).step_by(1.0))
                    .changed()
                {
                    self.s_controls
                        .send((DfControl::MaxDfThreshDb, self.max_dfthreshdb))
                        .ok();
                }
            }

            ui.label("Noise Attenuation [dB]");
            if ui
                .add(egui::Slider::new(&mut self.atten_lim, 0.0..=100.0))
                .changed()
            {
                self.s_controls
                    .send((DfControl::AttenLim, self.atten_lim))
                    .ok();
            }

            ui.label("Post Filter Beta");
            if ui
                .add(egui::Slider::new(&mut self.post_filter_beta, 0.0..=1.0).step_by(0.001))
                .changed()
            {
                self.s_controls
                    .send((DfControl::PostFilterBeta, self.post_filter_beta))
                    .ok();
            }

            ui.separator();

            // Spectrograms
            if self.df_worker.is_some() {
                if let Some(ref texture) = self.noisy_texture {
                    ui.label("Noisy");
                    ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(1000.0, 250.0)));
                }
                if let Some(ref texture) = self.enh_texture {
                    ui.label("DeepFilterNet Enhanced");
                    ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(1000.0, 250.0)));
                }
            }

            ui.separator();

            // SNR display
            ui.horizontal(|ui| {
                ui.label("Current SNR:");
                ui.label(format!("{:>5.1} dB", self.lsnr));
            });
        });
    }
}
