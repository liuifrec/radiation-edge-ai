"""Load the pretrained DNAi MobileOne-S1 model and run one GPU tensor probe.

This is intentionally a model/runtime smoke test rather than a biological
benchmark. It verifies that the pinned DNAi source revision can retrieve its
pretrained weights, move the model to CUDA, and execute a 1024x1024 FP32 tile.
"""

from __future__ import annotations

import time

import torch

from dnafiber.model.models_zoo import Models
from dnafiber.model.utils import _get_model


def main() -> int:
    print("Radiation Edge AI - DNAi MobileOne-S1 GPU Probe")
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the DNAi FP32 reference probe")

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    print("[Load pretrained model]")
    t0 = time.perf_counter()
    model = _get_model(Models.UNET_MOBILEONE_S1).to(device).eval()
    torch.cuda.synchronize()
    print(f" - Success ({time.perf_counter() - t0:.2f}s)")

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_params:,}")
    print(f"trainable parameters: {trainable:,}")

    # DNAi segmentation is 3-class: background, red, green.
    # Use a deterministic synthetic tensor only to verify the full model path.
    torch.manual_seed(0)
    x = torch.rand((1, 3, 1024, 1024), dtype=torch.float32, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print("[Inference]")
    # One warm-up pass avoids reporting first-kernel startup as steady-state time.
    with torch.inference_mode():
        _ = model(x)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        y = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    print(" - Success")
    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"output dtype: {y.dtype}")
    print(f"output device: {y.device}")
    print(f"latency (single 1024 tile, post-warmup): {elapsed * 1000:.2f} ms")
    print(
        "peak CUDA memory: "
        f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB"
    )

    if y.ndim != 4 or y.shape[0] != 1 or y.shape[1] != 3:
        raise SystemExit(
            f"Unexpected DNAi output shape {tuple(y.shape)}; expected N x 3 x H x W"
        )

    print("DNAI FP32 MODEL READY: YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
