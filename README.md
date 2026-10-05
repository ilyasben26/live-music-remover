

<div align="center">

<img src="assets/logo.svg" alt="Fast Music Remover Logo" width="100">

# Live Music Remover
</div>
<div align="center">

  [![GitHub release](https://img.shields.io/github/v/release/ilyasben26/live-music-remover?include_prereleases)](https://github.com/ilyasben26/live-music-remover/releases)
  [![GitHub license](https://img.shields.io/github/license/ilyasben26/live-music-remover)](https://github.com/ilyasben26/live-music-remover/blob/main/LICENSE)
  [![GitHub issues](https://img.shields.io/github/issues/ilyasben26/live-music-remover?color=blue)](https://github.com/ilyasben26/live-music-remover/issues)
  ![Platform](https://img.shields.io/badge/platform-Windows-blue?logo=windows)
  ![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/ilyasben26/live-music-remover/total)
 ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ilyasben26/live-music-remover/rust.yml)

  [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/A0A21O64X2)
</div>

A Windows app that removes background music and noise from your computer's audio in real time while keeping speech intact so you can watch videos without the music. It captures your system audio through a virtual audio cable ([VB-CABLE](https://vb-audio.com/Cable)), removes the music and plays the cleaned-up audio through your speakers or headphones. It can use [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance)'s denoiser (the default), [DPDFNet](https://github.com/ceva-ip/DPDFNet) or [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet).


> Very Important Note (PLEASE READ): In some cases, Live Music Remover can let through short blips of music or very faint musical artifacts, but this rarely happens and only when the music is too loud while a person is speaking. Furthermore, the blips don't form any coherent melody and are only perceptible when using headphones, if you notice them, try playing the audio through speakers, that should limit your perception of it. You have been warned. If I can, I will continue to monitor advances in this noise/music removal field and implement them on this project. If you know how to improve Live Music Remover, please open an issue or a pull request, any help will be appreciated. This project is a work in progress, there are bugs, if you encounter one, just close the app and open it again and that should fix it, and also make sure to open an issue on Github [here](https://github.com/ilyasben26/live-music-remover/issues) where you explain the bug so that I can fix it.


> Another Note: Live Music Remover currently supports Windows only. MacOS integration is planned. For Linux users who want to remove music in real-time from their system's audio, there is a native way to do it on Linux using a LADSPA plugin, see this [here](https://github.com/Rikorose/DeepFilterNet/blob/main/ladspa/filter-chain-configs/deepfilter-stereo-sink.conf). (A video tutorial on the Linux setup is coming soon)

## Demo
todo!

## Instructions

<!-- Watch the tutorial video: TODO!(insert video link) -->
<!-- todo!(add pictures to the steps below to make it clearer) -->
- To capture the system's audio and route it to *Live Music Remover*, install VB-CABLE (https://vb-audio.com/Cable).
- In the windows sound setting, select **CABLE Input (VB-Audio Virtual Cable)**.`
- Download the latest release of *Live Music Remover* from todo!(insert releases latest link)
- Launch the downloaded executable. You might get a Windows SmartScreen warning that it can harm your device, just click on *Run Anyway*. It's totally safe, you can read the code and compile the binary yourself if you don't trust me.


<!-- ## Features
- Live music/noise removal. -->

## Planned Features
- Removing music from user-provided video / audio files.
- Porting  to MacOS.
- Improving the music removal pipeline.
- Fixing bugs.

## Dev Setup
### Prerequisites
- [Rust](https://rust-lang.org/tools/install/)

### Running in dev mode
```powershell
git clone https://github.com/ilyasben26/live-music-remover
$Env:RUST_LOG="live-_music_remover=INFO,live_music_remover=DEBUG,df=DEBUG"
cargo +nightly build -p live-music-remover --features "ui,dev-console" --bin live-music-remover --release
```

## Credits / Citations


- The real-time audio processing code and the spectograms were initially copied from https://github.com/Rikorose/DeepFilterNet/tree/main/demo and slightly modified to support stereo audio and adjustable volume gain.
- DeepFilterNet Repo: https://github.com/Rikorose/DeepFilterNet
- DeepFilterNet citations:
  ```bibtex
  @inproceedings{schroeter2022deepfilternet,
    title={{DeepFilterNet}: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering}, 
    author = {Schröter, Hendrik and Escalante-B., Alberto N. and Rosenkranz, Tobias and Maier, Andreas},
    booktitle={ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year={2022},
    organization={IEEE}
  }
  ```
  ```bibtex
  @inproceedings{schroeter2023deepfilternet3,
    title = {{DeepFilterNet}: Perceptually Motivated Real-Time Speech Enhancement},
    author = {Schröter, Hendrik and Rosenkranz, Tobias and Escalante-B., Alberto N. and Maier, Andreas},
    booktitle={INTERSPEECH},
    year = {2023},
  }
  ```
- DPDFNet Repo: https://github.com/ceva-ip/DPDFNet (models: https://huggingface.co/Ceva-IP/DPDFNet, Apache-2.0). The DPDFNet-2 and DPDFNet-8 48 kHz HR models are bundled as ONNX, and the streaming code in `src/dpdfnet.rs` is a Rust port of the reference `StreamEnhancer`.
- DPDFNet citation:
  ```bibtex
  @article{rika2025dpdfnet,
    title  = {DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN},
    author = {Rika, Daniel and Sapir, Nino and Gus, Ido},
    journal = {arXiv preprint arXiv:2512.16420},
    year   = {2025}
  }
  ```
- Resemble Enhance Repo: https://github.com/resemble-ai/resemble-enhance (weights: https://huggingface.co/ResembleAI/resemble-enhance, MIT, © 2023 Resemble AI). Only the denoiser is used: it is exported from the `enhancer_stage2` checkpoint to ONNX with `scripts/export_resemble_enhance.py` and bundled as `models/resemble_denoiser.onnx`.
- DPDFNet and Resemble Enhance run on [ONNX Runtime](https://onnxruntime.ai) through the [`ort`](https://github.com/pykeio/ort) crate (Resemble Enhance uses its DirectML execution provider for the GPU).


