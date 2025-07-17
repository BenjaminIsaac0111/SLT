import json
import os

import numpy as np
import tensorflow as tf

try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError as e:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


def process_path_value_per_class(file_path=None, no_channels=None):
    xLeft, xRight = load_patch(file_path)

    # Extract the segmentation mask
    mask = xRight[:, :, 2]

    # Create one-hot encoded labels for values > 0
    one_hot = tf.one_hot(mask, depth=no_channels + 1, dtype=tf.float32)

    # Remove the extra channel for the 0 values
    channels = one_hot[:, :, 1:]

    return xLeft, channels


def load_patch(file_path=None):
    img = tf.io.read_file(file_path)
    x = decode_img(img)
    mid_point = tf.shape(x)[1] // 2
    xLeft = x[:, :mid_point, :]
    xRight = tf.cast(x[:, mid_point:, :] * 255, tf.int32)
    return xLeft, xRight


def decode_img(img):
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def get_dataset(
        cfg=None,
        repeat=True,
        transforms=None,
        filelists=None,  # Can be a single path (str) or a list of paths
        shuffle=True,
        batch_size=None,
):
    # Convert a single filelist path to a list if necessary
    if isinstance(filelists, str):
        filelists = [filelists]  # Wrap the single filelist path in a list

    all_files = []  # List to collect all file paths from the .txt files

    # Iterate over each .txt file in the list
    for filelist in filelists:
        with open(filelist, 'r') as f:
            # Read the file paths from the .txt file and prepend the directory from attention_unet.config
            files_in_list = [os.path.join(cfg['DATA_DIR'], line.strip()) for line in f.readlines()]
            files_in_list = [patch.split('\t')[0] for patch in files_in_list]
            all_files.extend(files_in_list)  # Add to the collective list of file paths

    # Create a dataset from the aggregated list of file paths
    ds = tf.data.Dataset.from_tensor_slices(all_files)

    if shuffle:
        # Shuffle using the provided buffer size in cfg
        ds = ds.shuffle(buffer_size=cfg['SHUFFLE_BUFFER_SIZE'])

    if repeat:
        ds = ds.repeat(-1)  # Repeat the dataset indefinitely

    def process_file(filepath):
        # Process each file path (loading, preprocessing, etc.)
        image, label = process_path_value_per_class(filepath, cfg['OUT_CHANNELS'])
        filename = tf.strings.split(filepath, os.sep)[-1]
        return filename, image, label

    # Apply processing function to each element in the dataset
    ds = ds.map(process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch the dataset
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)

    if transforms:
        # Wrap the transforms to ignore the filename
        def transform_with_filename(filename, image, label):
            image, label = transforms(image, label)
            return filename, image, label

        # Apply additional transformations if provided
        ds = ds.map(transform_with_filename, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetch data for optimized data loading
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def get_dataset_v2(
        data_dir,  # Path to the data directory
        filelists=None,  # Can be a single path (str) or a list of paths
        repeat=True,
        transforms=None,
        shuffle=True,
        batch_size=None,
        shuffle_buffer_size=256,  # Default shuffle buffer size, can be overridden
        out_channels=3  # Default number of output channels, can be overridden
):
    # Convert a single filelist path to a list if necessary
    if isinstance(filelists, str):
        filelists = [filelists]  # Wrap the single filelist path in a list

    all_files = []  # List to collect all file paths from the .txt files

    # Iterate over each .txt file in the list
    for filelist in filelists:
        with open(filelist, 'r') as f:
            # Read the file paths from the .txt file and prepend the directory from data_dir
            files_in_list = [os.path.join(data_dir, line.strip()) for line in f.readlines()]
            files_in_list = [patch.split('\t')[0] for patch in files_in_list]
            all_files.extend(files_in_list)  # Add to the collective list of file paths

    # Create a dataset from the aggregated list of file paths
    ds = tf.data.Dataset.from_tensor_slices(all_files)

    if shuffle:
        # Shuffle using the provided buffer size
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    if repeat:
        ds = ds.repeat(-1)  # Repeat the dataset indefinitely

    def process_file(filepath):
        # Process each file path (loading, preprocessing, etc.)
        image, label = process_path_value_per_class(filepath, out_channels)
        filename = tf.strings.split(filepath, os.sep)[-1]
        return filename, image, label

    # Apply processing function to each element in the dataset
    ds = ds.map(process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch the dataset
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)

    if transforms:
        # Wrap the transforms to ignore the filename
        def transform_with_filename(filename, image, label):
            image, label = transforms(image, label)
            return filename, image, label

        # Apply additional transformations if provided
        ds = ds.map(transform_with_filename, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetch data for optimized data loading
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def get_dataset_from_dir(
        cfg=None,
        repeat=True,
        transforms=None,
        shuffle=True,
        batch_size=None,
        directory=None
):
    filelist = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]

    # Create a dataset from the aggregated list of file paths
    ds = tf.data.Dataset.from_tensor_slices(filelist)

    if shuffle:
        # Shuffle using the provided buffer size in cfg
        ds = ds.shuffle(buffer_size=cfg['SHUFFLE_BUFFER_SIZE'])

    if repeat:
        ds = ds.repeat(-1)  # Repeat the dataset indefinitely

    def process_file(filepath):
        # Process each file path (loading, preprocessing, etc.)
        image, label = process_path_value_per_class(filepath, cfg['OUT_CHANNELS'])
        filename = tf.strings.split(filepath, os.sep)[-1]
        return filename, image, label

    # Apply processing function to each element in the dataset
    ds = ds.map(process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch the dataset
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)

    if transforms:
        # Wrap the transforms to ignore the filename
        def transform_with_filename(filename, image, label):
            image, label = transforms(image, label)
            return filename, image, label

        # Apply additional transformations if provided
        ds = ds.map(transform_with_filename, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetch data for optimized data loading
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def get_label_for_file(file_path):
    filename = os.path.basename(file_path)
    label = int(filename[-5])
    return label


# ------------------------------------------------------------------
# Helper: pure‑NumPy circle rasteriser (fast, avoids TF control‑flow)
# ------------------------------------------------------------------
def _draw_mask_np(coords, class_ids, height, width,
                  radius: int, out_channels: int):
    """
    Args
    ----
    coords       : (N, 2)  integer  [[x0, y0], …]  -- x ≡ cols, y ≡ rows
    class_ids    : (N,)    integer  class index *starting at 0*
    height, width: int
    radius       : int      circle radius in pixels
    out_channels : int      number of semantic classes

    Returns
    -------
    mask         : uint8  shape (H, W, C)   one‑hot (0/1)
    """
    yy, xx = np.ogrid[:height, :width]  # broadcasting grids
    mask = np.zeros((height, width, out_channels),
                    dtype=np.uint8)

    r2 = radius * radius
    for (x, y), cls in zip(coords, class_ids):
        d2 = (yy - y) ** 2 + (xx - x) ** 2  # squared distance
        circle = d2 <= r2
        mask[circle, cls] = 1  # write class plane
    return mask


# ------------------------------------------------------------------


def get_dataset_from_json(json_path,
                          images_dir,
                          *,
                          radius=8,
                          out_channels=9,
                          batch_size=1,
                          shuffle=True,
                          repeat=True,
                          transforms=None,
                          shuffle_buffer=256):
    """
    Parameters marked * are keyword‑only for clarity.
    -----------------------------------------------------------------
    json_path     : str   path to the annotation JSON (example you posted)
    images_dir    : str   directory that actually contains the *.png files
    radius        : int   circle radius for each point annotation
    out_channels  : int   number of semantic classes
    batch_size    : int
    shuffle       : bool  shuffle entire file list?
    repeat        : bool  repeat indefinitely?
    transforms    : callable(image, label) -> (image, label)  (optional)
    shuffle_buffer: int   TF shuffle buffer size
    """
    # 1) ----------------------------------------------------------------
    #    Load annotations once, keep a lightweight Python list of tuples
    # -------------------------------------------------------------------
    with open(json_path, "r") as f:
        ann = json.load(f)

    records = []  # [(filepath, coords[n,2], cls[n]), …]
    for fname, marks in ann.items():
        coords = np.array([m["coord"] for m in marks], dtype=np.int32)
        class_ids = np.array([m["class_id"] for m in marks], dtype=np.int32)
        # If class_id in JSON starts at 1, shift to 0‑based indexing:
        # class_ids -= 1
        records.append((os.path.join(images_dir, fname),
                        coords,
                        class_ids))

    # 2) ----------------------------------------------------------------
    #    Build a tf.data.Dataset whose *source* is the Python list above
    # -------------------------------------------------------------------
    ds = tf.data.Dataset.from_generator(
        lambda: records,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),  # file path
            tf.TensorSpec(shape=(None, 2), dtype=tf.int32),  # coords
            tf.TensorSpec(shape=(None,), dtype=tf.int32)))  # class_ids

    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    if repeat:
        ds = ds.repeat(-1)

    # 3) ----------------------------------------------------------------
    #    Map: read image → build mask → optional augment → return sample
    # -------------------------------------------------------------------
    def _process(filepath, coords, class_ids):
        img_bin = tf.io.read_file(filepath)
        img = tf.image.decode_png(img_bin, channels=3)  # float32 later
        h, w = tf.shape(img)[0], tf.shape(img)[1]

        # py_function: runs NumPy code above, avoids TF while‑loops
        mask = tf.py_function(
            func=lambda c, ids, H, W: _draw_mask_np(c, ids, H, W,
                                                    radius, out_channels),
            inp=[coords, class_ids, h, w],
            Tout=tf.uint8)
        mask.set_shape([None, None, out_channels])
        mask = tf.cast(mask, tf.float32)

        img = tf.image.convert_image_dtype(img, tf.float32)  # 0‑1 float

        if transforms is not None:
            img, mask = transforms(img, mask)

        filename = tf.strings.split(filepath, os.sep)[-1]
        return filename, img, mask

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
