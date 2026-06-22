"""
sr_realesrgan.py — GAN-based super-resolution (Real-ESRGAN).

Algorithm: adversarial generative upscaling. A generator network (RRDBNet),
trained against a discriminator, hallucinates plausible high-frequency detail.
Sharp/photo-real on faces. Pretrained — inference only.

Run:  python sr_realesrgan.py
Outputs -> data/sr/realesrgan/
"""

# --- basicsr / torchvision compatibility patch -----------------------------
# Newer torchvision removed `torchvision.transforms.functional_tensor`, which
# basicsr still imports. Alias it to the current module BEFORE importing basicsr.
import sys, types
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    import torchvision.transforms.functional as _F
    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim
# ---------------------------------------------------------------------------

import time
import urllib.request

import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

import config as C

# Official pretrained x4 weights (Real-ESRGAN general model).
WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
WEIGHTS_PATH = C.WEIGHTS_DIR / "RealESRGAN_x4plus.pth"


def ensure_weights():
    """Download the generator weights once into the shared weights cache."""
    if WEIGHTS_PATH.exists():
        return
    print(f"Downloading Real-ESRGAN weights -> {WEIGHTS_PATH.name}")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)


def build_upsampler() -> RealESRGANer:
    """Construct the RRDBNet generator and wrap it in the tiled inference helper."""
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=C.SCALE,
    )
    return RealESRGANer(
        scale=C.SCALE,
        model_path=str(WEIGHTS_PATH),
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=C.USE_FP16,
        device=C.DEVICE,
    )


def main():
    print(f"[Real-ESRGAN] {C.describe_env()}")
    ensure_weights()

    import kagglehub
    src = kagglehub.dataset_download(C.KAGGLE_SLUG)
    images = C.list_images(C.Path(src))
    if not images:
        raise SystemExit(f"No images found under {src}")
    print(f"Found {len(images)} image(s). Writing to {C.SR_REALESRGAN_DIR}")

    upsampler = build_upsampler()

    for i, path in enumerate(images, 1):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  skip (unreadable): {path.name}")
            continue
        t0 = time.time()
        output, _ = upsampler.enhance(img, outscale=C.SCALE)
        out_path = C.SR_REALESRGAN_DIR / f"{path.stem}_realesrgan_x{C.SCALE}.png"
        cv2.imwrite(str(out_path), output)
        print(f"  [{i}/{len(images)}] {path.name} "
              f"{img.shape[1]}x{img.shape[0]} -> {output.shape[1]}x{output.shape[0]} "
              f"({time.time()-t0:.1f}s)")

    print("Done (Real-ESRGAN).")


if __name__ == "__main__":
    main()