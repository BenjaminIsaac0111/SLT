import random
from collections import defaultdict

import os
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
            # Read the file paths from the .txt file and prepend the directory from cfg
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


def get_balanced_upsampled_dataset(cfg=None, repeat=True, transforms=None, filelists=None, shuffle=True,
                                   buffer_size=256, batch_size=None):
    # Convert a single filelist path to a list if necessary
    if isinstance(filelists, str):
        filelists = [filelists]  # Wrap the single filelist path in a list

    all_files = []  # List to collect all file paths from the .txt files
    label_file_dict = defaultdict(list)  # Dictionary to hold files by label

    # Iterate over each .txt file in the list
    for filelist in filelists:
        with open(filelist, 'r') as f:
            # Read the file paths from the .txt file and prepend the directory from cfg
            files_in_list = [os.path.join(cfg['DATA_DIR'], line.strip()) for line in f.readlines()]
            files_in_list = [patch.split('\t')[0] for patch in files_in_list]
            for file_path in files_in_list:
                label = get_label_for_file(file_path)
                label_file_dict[label].append(file_path)
            all_files.extend(files_in_list)  # Add to the collective list of file paths

    # Find the maximum class count for upsampling
    max_class_count = max(len(files) for files in label_file_dict.values())

    # Upsample minority classes
    upsampled_files = []
    for label, files in label_file_dict.items():
        upsampled_files.extend(files * (max_class_count // len(files)) + files[:max_class_count % len(files)])

    # Create a dataset from the upsampled list of file paths
    ds = tf.data.Dataset.from_tensor_slices(upsampled_files)

    if shuffle:
        # Shuffle the dataset
        ds = ds.shuffle(buffer_size=buffer_size)

    if repeat:
        ds = ds.repeat(-1)  # Repeat the dataset indefinitely

    def process_file(filepath):
        # Process each file path (loading, preprocessing, etc.)
        image, label = process_path_value_per_class(filepath, cfg['OUT_CHANNELS'])
        filename = tf.strings.split(filepath, os.sep)[-1]
        return filename, image, label

    # Apply processing function to each element in the dataset
    ds = ds.map(process_file, num_parallel_calls=AUTOTUNE)

    # Batch the dataset and randomly sample a subsample in each batch
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    ds = ds.map(lambda filenames, images, labels: (filenames, images, labels), num_parallel_calls=AUTOTUNE)

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


def get_label_for_file(file_path):
    filename = os.path.basename(file_path)
    label = int(filename[-5])
    return label
