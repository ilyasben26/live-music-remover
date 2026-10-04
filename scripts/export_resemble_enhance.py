"""Export the Resemble Enhance denoiser (https://github.com/resemble-ai/resemble-enhance)
to ONNX for the live pipeline in src/resemble.rs.

Usage:
    git clone https://github.com/resemble-ai/resemble-enhance
    pip install -r scripts/requirements-resemble-export.txt
    python scripts/export_resemble_enhance.py --repo resemble-enhance --out models

Produces models/resemble_denoiser.onnx, a fixed-length graph
(N = 420 * K samples at 44.1 kHz, K = --frames):

    wav (1, N) -> wav_out (1, N)

It takes raw (unnormalised) audio and returns audio at the input's scale. The
weights come from the enhancer_stage2 checkpoint, which includes the denoiser.

Export notes:
- STFT / ISTFT are written as Conv / MatMul so the graph only uses plain ops.
  A large-kernel ConvTranspose ISTFT is very slow on DirectML.
- Large weights are stored as float16 with a Cast to float32 (compute stays
  float32), which halves the file size.
"""
import argparse
import hashlib
import math
import sys
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HOP = 420
HF_BASE = "https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/"
CKPT_FILES = ["hparams.yaml", "ds/G/latest", "ds/G/default/mp_rank_00_model_states.pt"]


# ── Loading ───────────────────────────────────────────────────────────────────


def import_resemble(repo: Path):
    sys.path.insert(0, str(repo))
    # Training-only dependencies pulled in by imports; not needed for inference.
    for m in [
        "deepspeed",
        "deepspeed.accelerator",
        "deepspeed.runtime",
        "deepspeed.runtime.engine",
        "deepspeed.runtime.utils",
        "celluloid",
        "ptflops",
        "librosa",
        "resampy",
    ]:
        sys.modules.setdefault(m, MagicMock())
    if sys.platform == "win32":
        import pathlib

        pathlib.PosixPath = pathlib.WindowsPath  # hparams.yaml pickles PosixPath


def download_checkpoint(ckpt: Path):
    for rel in CKPT_FILES:
        path = ckpt / rel
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", rel)
        urllib.request.urlretrieve(HF_BASE + rel + "?download=true", path)


def load_enhancer(ckpt: Path):
    from resemble_enhance.enhancer.enhancer import Enhancer
    from resemble_enhance.enhancer.hparams import HParams
    from resemble_enhance.inference import remove_weight_norm_recursively

    hp = HParams.load(ckpt)
    enh = Enhancer(hp)
    sd = torch.load(ckpt / CKPT_FILES[2], map_location="cpu", weights_only=False)["module"]
    enh.load_state_dict(sd)
    enh.eval()
    remove_weight_norm_recursively(enh)
    return enh


# ── Signal processing as plain ops ────────────────────────────────────────────


def hann(n_fft):
    return torch.hann_window(n_fft, periodic=True, dtype=torch.float64)


class Stft(nn.Module):
    """torch.stft(center=True, window=hann) as a Conv; returns (re, im), each (B, F, T+1)."""

    def __init__(self, n_fft, hop, pad_mode):
        super().__init__()
        self.n_fft, self.hop, self.pad_mode = n_fft, hop, pad_mode
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        ang = 2 * math.pi * k * n / n_fft
        w = hann(n_fft)
        basis = torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w])
        self.register_buffer("basis", basis[:, None, :].float())

    def forward(self, x):  # (B, N)
        p = self.n_fft // 2
        x = F.pad(x[:, None], (p, p), mode=self.pad_mode)
        s = F.conv1d(x, self.basis, stride=self.hop)
        nf = self.n_fft // 2 + 1
        return s[:, :nf], s[:, nf:]


class Istft(nn.Module):
    """torch.istft(center=True, window=hann) for a fixed frame count, as MatMul + overlap-add."""

    def __init__(self, n_fft, hop, n_frames):
        super().__init__()
        assert n_fft % hop == 0
        self.n_fft, self.hop = n_fft, hop
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        c = torch.full_like(k, 2.0)
        c[0] = c[-1] = 1.0
        ang = 2 * math.pi * k * n / n_fft
        w = hann(n_fft)
        basis = torch.cat([c * torch.cos(ang) * w, -c * torch.sin(ang) * w]) / n_fft
        self.register_buffer("basis_t", basis.T.contiguous().float())  # (n_fft, 2F)
        env = F.conv_transpose1d(
            torch.ones(1, 1, n_frames, dtype=torch.float64), (w * w)[None, None], stride=hop
        )
        self.register_buffer("inv_env", (1.0 / env.clamp_min(1e-11)).float())

    def forward(self, re, im, length):
        s = torch.cat([re, im], dim=1)  # (B, 2F, T)
        frames = torch.matmul(self.basis_t, s)  # (B, n_fft, T)
        B, _, T = frames.shape
        r = self.n_fft // self.hop
        seg = frames.reshape(B, r, self.hop, T).permute(0, 1, 3, 2)  # (B, r, T, hop)
        y = sum(F.pad(seg[:, j], (0, 0, j, r - 1 - j)) for j in range(r))  # (B, T+r-1, hop)
        y = y.reshape(B, 1, -1) * self.inv_env
        p = self.n_fft // 2
        return y[:, 0, p : p + length]


# ── Exported graphs ───────────────────────────────────────────────────────────


class DenoiserOnnx(nn.Module):
    def __init__(self, denoiser, n_samples):
        super().__init__()
        self.d = denoiser
        n_fft = denoiser.stft_cfg["n_fft"]
        self.n = n_samples
        self.stft = Stft(n_fft, HOP, "reflect")
        self.istft = Istft(n_fft, HOP, n_samples // HOP + 1)

    def forward(self, x):
        peak = x.abs().amax(dim=-1, keepdim=True) + 1e-7
        re, im = self.stft(x / peak)
        re, im = re[..., :-1], im[..., :-1]
        mag = (re * re + im * im).sqrt()
        phi = torch.atan2(im, re)
        cos, sin = phi.cos(), phi.sin()
        mag_mask, sin_res, cos_res = self.d._predict(mag, cos, sin)
        sm, sc, ss = self.d._separate(mag, cos, sin, mag_mask, cos_res, sin_res)
        re_o, im_o = sm * sc, sm * ss
        re_o = torch.cat([re_o, re_o[..., -1:]], dim=-1)
        im_o = torch.cat([im_o, im_o[..., -1:]], dim=-1)
        return self.istft(re_o, im_o, self.n) * peak


# ── ONNX post-processing ──────────────────────────────────────────────────────


def fp16_storage(model):
    """Store large float32 initializers as float16 + Cast(to=float)."""
    from onnx import TensorProto, helper, numpy_helper

    g = model.graph
    inits, casts = [], []
    for init in g.initializer:
        a = numpy_helper.to_array(init) if init.data_type == TensorProto.FLOAT else None
        if a is not None and a.size >= 1024 and np.abs(a).max() < 60000:
            h = numpy_helper.from_array(a.astype(np.float16), init.name + "__fp16")
            inits.append(h)
            casts.append(
                helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast")
            )
        else:
            inits.append(init)
    nodes = list(g.node)
    del g.initializer[:]
    g.initializer.extend(inits)
    del g.node[:]
    g.node.extend(casts + nodes)
    return model


def export(model, inputs, names, path, meta):
    import onnx
    import onnxslim

    torch.onnx.export(
        model, inputs, str(path), input_names=names, output_names=["wav_out"], opset_version=17, dynamo=False
    )
    m = fp16_storage(onnxslim.slim(onnx.load(str(path))))
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path} ({path.stat().st_size} bytes, sha256 {digest})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True, help="clone of resemble-ai/resemble-enhance")
    ap.add_argument("--ckpt", type=Path, default=None, help="enhancer_stage2 dir (downloaded if missing)")
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--frames", type=int, default=128, help="STFT frames per chunk, multiple of 16")
    args = ap.parse_args()
    assert args.frames % 16 == 0

    import_resemble(args.repo)
    ckpt = args.ckpt or Path("enhancer_stage2")
    download_checkpoint(ckpt)
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    N = args.frames * HOP
    denoiser = load_enhancer(ckpt).denoiser
    graph = DenoiserOnnx(denoiser, N).eval()

    x = torch.randn(1, N) * 0.1
    with torch.inference_mode():
        peak = x.abs().max()
        ref = denoiser(x / peak) * peak
        y = graph(x)
        print("graph vs upstream, max abs err:", (ref - y).abs().max().item())

    meta = {
        "sample_rate": "44100",
        "hop_length": str(HOP),
        "chunk_frames": str(args.frames),
        "chunk_samples": str(N),
        "source": "https://huggingface.co/ResembleAI/resemble-enhance (enhancer_stage2 denoiser, MIT)",
    }
    path = args.out / "resemble_denoiser.onnx"
    with torch.inference_mode():
        export(graph, (x,), ["wav"], path, meta)

    import onnxruntime as ort

    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"wav": x.numpy()})[0]
    snr = 10 * np.log10((y.numpy() ** 2).sum() / ((y.numpy() - got) ** 2).sum())
    print(f"onnx vs torch: {snr:.1f} dB SNR")


if __name__ == "__main__":
    main()
