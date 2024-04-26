import os
import tensorflow as tf

try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError as e:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


def process_path_value_per_class(file_path=None, no_channels=None):
    xLeft, xRight = load_patch(file_path)
    c = 1
    output_list = []
    while c <= no_channels:
        masked = xRight[:, :, 2] == c
        output_list.append(masked)
        c = c + 1
    channels = tf.cast(tf.stack(output_list, axis=2), tf.float32)
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
        return process_path_value_per_class(filepath, cfg['OUT_CHANNELS'])

    # Apply processing function to each element in the dataset
    ds = ds.map(process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch the dataset
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)

    if transforms:
        # Apply additional transformations if provided
        ds = ds.map(transforms, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Prefetch data for optimized data loading
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

