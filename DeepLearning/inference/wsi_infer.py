#!/usr/bin/env python3
"""
minimal_wsi_infer.py  (with jet colormap for uncertainty)
---------------------------------------------
Runs a TensorFlow-2 / `tf.keras` segmentation model over one `.svs` slide
using GPU mixed precision, blending overlapping tiles, and writing a
multi-resolution OME-TIFF. Uses on-disk memmaps for accumulators to reduce
RAM usage. Outputs uncertainty (entropy) map with a Jet colormap.
"""

# -----------------------------------------------------------------------------
# ⇨⇨⇨ EDIT THESE CONSTANTS ⇦⇦⇦
# -----------------------------------------------------------------------------
SLIDE_PATH = r"C:\Users\wispy\OneDrive - University of Leeds\DATABACKUP\Clasicc\21043\21074.svs"  # ← CHANGE ME
MODEL_PATH = (r"C:\Users\wispy\OneDrive - University of Leeds\PhD "
              r"Projects\Attention-UNET\cfg\unet_training_experiments\Outputs\dropout_attention_unet_fl_f1.h5"
              r"\best_dropout_attention_unet_fl_f1.h5")  # ← CHANGE ME
OUTPUT_MASK = "slide_output_colored.ome.tiff"  # ← Now colored uncertainty
BATCH_SIZE = 1
OVERLAP_PX = 32  # should be multiple of model stride
MC_SAMPLES = 10  # Monte Carlo dropout samples
MODE = 'bald'  # 'mask' or 'bald'
# -----------------------------------------------------------------------------

import math
import os
import tempfile

import large_image as li
import matplotlib.cm as cm  # for Jet colormap
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tqdm import tqdm

# Enable mixed precision & XLA
try:
    mixed_precision.set_global_policy('mixed_float16')
    print('✓ Mixed precision enabled')
except Exception:
    print('! Mixed precision unavailable')
try:
    tf.config.optimizer.set_jit(True)
    print('✓ XLA JIT enabled')
except Exception:
    print('! XLA JIT unavailable')

# Custom objects
from DeepLearning.models.custom_layers import (
    DropoutAttentionBlock, GroupNormalization, SpatialConcreteDropout
)

CUSTOM_OBJECTS = {
    'DropoutAttentionBlock': DropoutAttentionBlock,
    'GroupNormalization': GroupNormalization,
    'SpatialConcreteDropout': SpatialConcreteDropout,
}


# Helpers

def load_model(path: str):
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)


def deduce_tile_size(model, fallback=512):
    _, H, W, _ = model.input_shape
    stride = 1
    for lyr in model.layers:
        st = getattr(lyr, 'strides', None)
        if st:
            s = st[0] if isinstance(st, (tuple, list)) else st
            stride *= s or 1

    def choose(dim):
        return int(dim) if dim else int(math.ceil(max(fallback, stride) / stride) * stride)

    return choose(H), choose(W), stride


def make_blend_weights(h, w, overlap):
    c = overlap // 2
    ramp = np.linspace(0, 1, c, dtype=np.float32)
    y = np.ones(h, dtype=np.float32)
    x = np.ones(w, dtype=np.float32)
    y[:c], y[-c:] = ramp, ramp[::-1]
    x[:c], x[-c:] = ramp, ramp[::-1]
    return np.outer(y, x)


# Batch inference + blending

def _run_batch(tiles, coords, infer_fn, accum, weight_acc, wmask, tile_hw):
    """Infer with MC dropout, blend overlapping tiles into accum buffers."""
    th, tw = tile_hw
    batch = tf.convert_to_tensor(np.stack(tiles), dtype=tf.float16)
    sum_prob = None
    for _ in range(MC_SAMPLES):
        probs = infer_fn(batch)
        sum_prob = probs if sum_prob is None else sum_prob + probs
    mean_prob = sum_prob / MC_SAMPLES

    for (x0, y0), mp in zip(coords, mean_prob):
        x0 = int(round(x0))
        y0 = int(round(y0))
        # full-tile prediction
        if MODE == 'mask':
            pm = np.argmax(mp, axis=-1).astype(np.float32)
        else:
            p = np.clip(mp, 1e-12, 1.0)
            pm = -np.sum(p * np.log(p), axis=-1).astype(np.float32)
        # compute actual patch size at edges
        h = min(th, accum.shape[0] - y0)
        w = min(tw, accum.shape[1] - x0)
        ys = slice(y0, y0 + h)
        xs = slice(x0, x0 + w)
        # crop prediction and weights to patch
        pm0 = pm[:h, :w]
        w0 = wmask[:h, :w]
        accum[ys, xs] += pm0 * w0
        weight_acc[ys, xs] += w0


# Main

tmp = tempfile.gettempdir()
print(f'Using temp dir {tmp}')


def infer_slide():
    model = load_model(MODEL_PATH)
    th, tw, _ = deduce_tile_size(model)
    overlap = max(OVERLAP_PX, _)
    infer_fn = tf.function(lambda x: model(x, training=(MODE == 'bald')), jit_compile=True)

    src = li.getTileSource(SLIDE_PATH)
    H_native, W_native = src.sizeY, src.sizeX

    sample_tile = next(src.tileIterator(tile_size={'width': tw, 'height': th}))
    scale = sample_tile['width'] // sample_tile['tile'].shape[1]

    # compute down-sampled slide dims at 10×
    H10, W10 = H_native // scale, W_native // scale

    # on-disk accumulators at 10× resolution
    accu_file = os.path.join(tmp, 'accu.npy')
    weig_file = os.path.join(tmp, 'weig.npy')
    accum = np.memmap(accu_file, dtype=np.float32, mode='w+', shape=(H10, W10))
    weight_acc = np.memmap(weig_file, dtype=np.float32, mode='w+', shape=(H10, W10))

    # precompute blend mask once (size th×tw at 10×)
    wmask = make_blend_weights(th, tw, overlap)

    tile_kw = {
        'tile_size': {'width': tw, 'height': th},
        'overlap': overlap,
        'format': li.constants.TILE_FORMAT_NUMPY,
    }

    for tile in tqdm(src.tileIterator(**tile_kw), desc='Tiles'):
        img = tile['tile'][..., :3].astype(np.float32) / 255.
        h, w, _ = img.shape
        # th, tw are your model’s expected spatial dims
        pad_h = th - h
        pad_w = tw - w
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img,
                         ((0, pad_h),
                          (0, pad_w),
                          (0, 0)),
                         mode='constant',
                         constant_values=0)
        coords = (tile['x'] // scale, tile['y'] // scale)  # now in 10× pixels

        # run MC-dropout & blending into 10× accumulators
        _run_batch(
            tiles=[img], coords=[coords],
            infer_fn=infer_fn,
            accum=accum, weight_acc=weight_acc,
            wmask=wmask, tile_hw=(th, tw)
        )

    # normalize and flush
    out10 = np.divide(accum, weight_acc, out=accum, where=weight_acc > 0)
    accum.flush()
    weight_acc.flush()

    if MODE == 'mask':
        # class mask as before
        out10 = np.rint(out10).astype(np.uint8)
        photometric = 'minisblack'
        data = out10
        axes = 'YX'
    else:
        # apply jet colormap to uncertainty map
        min_val, max_val = out10.min(), out10.max()
        norm = (out10 - min_val) / (max_val - min_val + 1e-12)
        jet_map = cm.get_cmap('jet')
        colored = jet_map(norm)[..., :3]  # (H,W,3) floats
        data = (colored * 255).astype(np.uint8)
        photometric = 'rgb'
        axes = 'YXC'

    # write out as OME-TIFF
    from tifffile import TiffWriter
    with TiffWriter(OUTPUT_MASK, bigtiff=True) as twr:
        twr.write(data,
                  photometric=photometric,
                  compression='zlib',
                  metadata={'axes': axes})


if __name__ == '__main__':
    infer_slide()
