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
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat(-1)  # Repeat the dataset indefinitely

    def process_file(filepath):
        # Process each file path (loading, preprocessing, etc.)
        image, label = process_path_value_per_class(filepath, out_channels)
        filename = tf.strings.split(filepath, os.sep)[-1]
        return filename, image, label

    # Apply processing function to each element in the dataset
    ds = ds.map(process_file, num_parallel_calls=AUTOTUNE)

    # Batch the dataset
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)

    if transforms:
        # Wrap the transforms to ignore the filename
        def transform_with_filename(filename, image, label):
            image, label = transforms(image, label)
            return filename, image, label

        # Apply additional transformations if provided
        ds = ds.map(transform_with_filename, num_parallel_calls=AUTOTUNE)

    # Prefetch data for optimized data loading
    ds = ds.prefetch(AUTOTUNE)

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


def get_dataset_from_dir_v2(
        repeat=True,
        transforms=None,
        shuffle_buffer_size=True,
        batch_size=None,
        images_dir=None
):
    filelist = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')]

    # Create a dataset from the aggregated list of file paths
    ds = tf.data.Dataset.from_tensor_slices(filelist)

    if shuffle_buffer_size:
        # Shuffle using the provided buffer size in cfg
        ds = ds.shuffle(shuffle_buffer_size)

    if repeat:
        ds = ds.repeat(-1)  # Repeat the dataset indefinitely

    def process_file(filepath):
        # Process each file path (loading, preprocessing, etc.)
        image, label = process_path_value_per_class(filepath, 9)
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


def _draw_mask_tf(coords, class_ids, height, width, radius, out_channels):
    ys = tf.cast(coords[:, 0], tf.int32)
    xs = tf.cast(coords[:, 1], tf.int32)

    yy = tf.range(height)[:, None]  # [H,1]
    xx = tf.range(width)[None, :]  # [1,W]

    dy = yy[None, :, :] - ys[:, None, None]  # [P,H,W]
    dx = xx[None, :, :] - xs[:, None, None]  # [P,H,W]
    circle = (dy * dy + dx * dx) <= radius * radius

    cls_oh = tf.one_hot(class_ids, out_channels, dtype=tf.float32)
    cls_oh = tf.reshape(cls_oh, [-1, 1, 1, out_channels])

    per_point = tf.cast(circle[..., None], tf.float32) * cls_oh

    mask = tf.reduce_max(per_point, axis=0)

    return mask


def get_dataset_from_json_v2(
        json_path,
        images_dir,
        *,
        radius=8,
        out_channels=9,
        batch_size=1,
        shuffle=True,
        repeat=True,
        transforms=None,
        shuffle_buffer=128
):
    # 1) Load annotations
    with open(json_path, "r") as f:
        ann = json.load(f)

    records = []
    for fname, marks in ann.items():
        coords = np.array([m["coord"] for m in marks], dtype=np.int32)
        class_ids = np.array([m["class_id"] for m in marks], dtype=np.int32)
        records.append((os.path.join(images_dir, fname), coords, class_ids))

    # 2) Create dataset of raw records
    ds = tf.data.Dataset.from_generator(
        lambda: records,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
    )
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat(-1)

    @tf.function
    def _process(filepath, coords, class_ids):
        with tf.device('/CPU:0'):
            # decode, draw mask, cast to float
            img = tf.image.decode_png(tf.io.read_file(filepath), channels=3)
            h, w = tf.shape(img)[0], tf.shape(img)[1]
            mask = _draw_mask_tf(coords, class_ids, h, w, radius, out_channels)
            half_w = w // 2
            img = img[:, :half_w, :]
            mask = mask[:, :half_w, :]

            img = tf.image.convert_image_dtype(img, tf.float32)
        return img, mask

    ds = ds.map(_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True)

    if transforms:
        def _transform_fn(imgs, masks):
            imgs, masks = transforms(imgs, masks)
            return imgs, masks

        ds = ds.map(_transform_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
