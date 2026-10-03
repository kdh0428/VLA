"""Image preprocessing of the OpenVLA LIBERO evaluation without TensorFlow (numpy + PIL only, used by simulator
workers so they do not import torch). See openvla_core.py for the replicated steps."""
import io

import numpy as np
from PIL import Image


def tf_crop_and_resize(img: np.ndarray, scale: float = 0.9, out: int = 224) -> np.ndarray:
    """tf.image.crop_and_resize (bilinear) of the centered box with area `scale`, float32 in [0,1] -> uint8."""
    H, W = img.shape[:2]
    s = np.sqrt(scale); y1 = x1 = (1 - s) / 2; y2 = x2 = y1 + s
    ys = y1 * (H - 1) + np.arange(out) * (y2 - y1) * (H - 1) / (out - 1)
    xs = x1 * (W - 1) + np.arange(out) * (x2 - x1) * (W - 1) / (out - 1)
    f = img.astype(np.float32) / 255.0
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1i = np.minimum(y0 + 1, H - 1); x1i = np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[:, None, None]; wx = (xs - x0)[None, :, None]
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1i] * wx
    bot = f[y1i][:, x0] * (1 - wx) + f[y1i][:, x1i] * wx
    r = top * (1 - wy) + bot * wy
    return (np.clip(r, 0, 1) * 255.0 + 0.5).astype(np.uint8)       # convert_image_dtype(saturate=True) rounds


def preprocess(agentview: np.ndarray) -> Image.Image:
    img = agentview[::-1, ::-1]
    buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(img)).save(buf, format="JPEG", quality=95); buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((224, 224), Image.LANCZOS)
    return Image.fromarray(tf_crop_and_resize(np.asarray(img)))
