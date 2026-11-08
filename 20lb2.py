"""Day 20 / Session 1 - Lab Assignment 2: U-Net Encoder-Decoder Shape Trace - Pool, Transposed-Conv & Skip Concat."""
import numpy as np


def same_conv(x, out_ch):
    x = np.asarray(x, dtype=float)
    m = x.mean(axis=0)                                  # per-pixel channel mean -> (H, W)
    return np.repeat(m[np.newaxis, :, :], out_ch, axis=0)   # broadcast to out_ch channels


def maxpool2x2(x):
    x = np.asarray(x, dtype=float)
    C, H, W = x.shape
    if H % 2 or W % 2:
        raise ValueError("height and width must be even for a 2x2 max-pool")
    blocks = x.reshape(C, H // 2, 2, W // 2, 2)          # tile into non-overlapping 2x2 blocks
    return blocks.max(axis=(2, 4))                       # a real block maximum


def upsample2x(x):
    x = np.asarray(x, dtype=float)
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)  # every pixel becomes a 2x2 block


def concat_skip(decoder, encoder):
    decoder = np.asarray(decoder)
    encoder = np.asarray(encoder)
    if decoder.shape[1:] != encoder.shape[1:]:
        raise ValueError("spatial dims must match to concatenate a skip connection")
    return np.concatenate([decoder, encoder], axis=0)    # channels add


def down_up_with_skip(x, mid_ch, out_ch):
    skip = same_conv(x, mid_ch)
    pooled = maxpool2x2(skip)
    bottleneck = same_conv(pooled, out_ch)
    up = same_conv(upsample2x(bottleneck), mid_ch)
    merged = concat_skip(up, skip)
    final = same_conv(merged, mid_ch)
    return {"skip": skip, "pooled": pooled, "bottleneck": bottleneck, "up": up, "merged": merged, "final": final}


if __name__ == "__main__":
    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]], dtype=float)
    print(maxpool2x2(x))                                                       # [[[ 6.  8.] [14. 16.]]]
    print(upsample2x(np.array([[[1, 2], [3, 4]]], dtype=float)).shape)         # (1, 4, 4)
    x = np.random.RandomState(0).rand(1, 64, 64)
    tr = down_up_with_skip(x, 8, 16)
    print({k: v.shape for k, v in tr.items()})
    print(np.array_equal(tr["merged"][:8], tr["up"]), np.array_equal(tr["merged"][8:], tr["skip"]))   # True True
    try:
        concat_skip(np.zeros((2, 4, 4)), np.zeros((2, 2, 2)))
    except ValueError:
        print("ValueError ok")
    try:
        maxpool2x2(np.zeros((1, 3, 4)))
    except ValueError:
        print("odd size ValueError ok")
    block = np.zeros((1, 2, 2)); block[0, 1, 1] = 9
    print(maxpool2x2(block)[0, 0, 0])                                          # 9.0 (block max, not top-left)
