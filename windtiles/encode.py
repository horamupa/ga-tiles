"""PNG wind tile: float header in row 0, u v T as bytes in RGB, byte 0 for no data."""

import io
import struct

import numpy as np
from PIL import Image

from windtiles.grid import NODES

HEADER = "<6f"  # uMin, uMax, vMin, vMax, tMin, tMax


def encode_tile(u: np.ndarray, v: np.ndarray, t: np.ndarray, per_channel: bool = False) -> bytes:
    """(181, 181) u, v (kt) and T (°C), row 0 southern, NaN for no data → PNG bytes. A NaN node is byte 0 in all
    three channels; with `per_channel` only in the channel that has the NaN."""
    fields = np.stack([u, v, t], axis=-1).astype(np.float64)
    mask = np.isnan(fields)
    if not per_channel:
        mask = np.broadcast_to(mask.any(axis=-1, keepdims=True), mask.shape)
    lows = np.where(mask, np.inf, fields).min(axis=(0, 1))
    highs = np.where(mask, -np.inf, fields).max(axis=(0, 1))
    empty = mask.all(axis=(0, 1))
    lows[empty] = highs[empty] = 0
    spans = np.where(highs > lows, highs - lows, 1.0)
    scaled = (fields - lows) / spans * 254
    nodes = np.where(mask, 0, 1 + np.rint(scaled)).astype(np.uint8)

    pixels = np.zeros((NODES + 1, NODES, 3), dtype=np.uint8)
    pixels[1:] = nodes[::-1]
    header = struct.pack(HEADER, *(x for pair in zip(lows, highs) for x in pair))
    pixels[0, :8] = np.frombuffer(header, np.uint8).reshape(8, 3)

    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def decode_tile(data: bytes, per_channel: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PNG bytes → (181, 181) float64 u, v, T, row 0 southern; NaN in all three where any byte is 0, with
    `per_channel` only in the channel whose byte is 0."""
    pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    header = struct.unpack(HEADER, bytes(pixels[0, :8].reshape(-1)))
    lows, highs = np.array(header[0::2]), np.array(header[1::2])
    nodes = pixels[1:][::-1]
    values = lows + (nodes - 1.0) * ((highs - lows) / 254)
    values[(nodes == 0) if per_channel else (nodes == 0).any(axis=-1)] = np.nan
    return values[..., 0], values[..., 1], values[..., 2]
