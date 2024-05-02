import os

from Model.custom_layers import AttentionBlock
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision

from TeaPotts.dataloader import get_dataset
from TeaPotts.transforms import Transforms
from TeaPotts.custom_layers import PixelShuffle
from TeaPotts.utils import build_seg_cmap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mixed_precision.set_global_policy('mixed_float16')

psuedo_working_dir = "Attention-UNET/"
cfg = {
    "MODEL_DIR": ("%sModel_Example_Outputs/" % psuedo_working_dir),
    "MODEL_NAME": "student_attention_resnet.h5",
    "N_MODEL_LEVELS": 2,  # Number of Down-sampling and Up-sampling levels
    "N_CONV_PER_LAYER": 2,  # Number of Conv blocks per level
    "USE_ATTENTION": True,  # Build model with additive attention block
    "N_FILTERS": 64,  # Filters per layers (This is constant, unlike the original UNET)
    "INPUT_SIZE": [1024, 512],  # Height, Width
    "OUT_CHANNELS": 9,  # Number of output classes
    "LEARNING_RATE": 1.0e-05,
    "USE_XLA": False,  # Enable Just in Time (JIT) compilation
    "DATA_DIR": ("%sExamples/" % psuedo_working_dir),  # Location of your data
    "TRAINING_LIST": ("%sExamples/train.txt" % psuedo_working_dir),
    # A .txt containing a list of filenames for training data
    "TESTING_LIST": ("%sExamples/test.txt" % psuedo_working_dir),
    # A .txt containing a list of filenames for test data used in validation
    "SHUFFLE_BUFFER_SIZE": 256,
    "BATCH_SIZE": 1,
    "EPOCHS": 10,
    "STEPS": 237,
    "VAL_BATCH_SIZE": 1,
    "VAL_STEPS": 60,
    "CLASS_COMPONENTS": {
        0: "Non-Informative",
        1: "Tumour",
        2: "Stroma",
        3: "Necrosis",
        4: "Vessel",
        5: "Inflammation",
        6: "Tumour-Lumen",
        7: "Mucin",
        8: "Muscle"
    },
    "USE_FOCAL_LOSS": True,  # else use weighted cross entropy
    "CLASS_WEIGHTS": [
        1.38333643,
        0.33804931,
        0.28400663,
        1.63984806,
        5.28325519,
        10.0,
        1.61236405,
        8.36243097,
        4.21306924
    ]
}

checkpoint_dir = cfg['MODEL_DIR']
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(f'{checkpoint_dir}/outputs/'):
    os.makedirs(f'{checkpoint_dir}/outputs/')

transforms = Transforms()
training_ds = get_dataset(
    cfg=cfg,
    repeat=True,
    shuffle=True,
    transforms=transforms,
    filelists=cfg['TRAINING_LIST'],
    batch_size=cfg['BATCH_SIZE']
)

# Should not include Transforms
memory_bank_ds = get_dataset(
    cfg=cfg,
    repeat=False,
    shuffle=False,
    filelists=cfg['TRAINING_LIST'],
    transforms=None,
    batch_size=1
)

test_ds = get_dataset(
    cfg=cfg,
    repeat=False,
    shuffle=False,
    filelists=cfg['TESTING_LIST'],
    transforms=None,
    batch_size=1
)


# %%
def plot_losses(log_file_path):
    # Define column names
    column_names = ['Epoch', 'Total Loss', 'Hard Label Loss', 'Soft Target Loss', 'Validation Loss']

    # Read the CSV file into a DataFrame with no header and assign column names
    data = pd.read_csv(log_file_path, header=None, names=column_names)

    # Check if the DataFrame is not empty
    if not data.empty:
        # Set the index to the 'Epoch' column for easier plotting
        data.set_index('Epoch', inplace=True)

        # Plotting each column
        plt.figure(figsize=(10, 8))
        for column in data.columns:
            plt.plot(data.index, data[column], label=column)

        # Adding titles and labels
        plt.title('Training and Validation Losses Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show()
    else:
        print("The data file is empty or not found.")


plot_losses(f"{cfg['MODEL_DIR']}/{cfg['MODEL_NAME']}/errors_{cfg['MODEL_NAME'][:-3]}.csv")

# %%
checkpoint_dir = f"{cfg['MODEL_DIR']}/{cfg['MODEL_NAME']}/ckpt_{cfg['MODEL_NAME']}"  # this is weird...
model = load_model(
    checkpoint_dir,
    custom_objects={
        'GroupNormalization': GroupNormalization(),
        'PixelShuffle': PixelShuffle,
        'AttentionBlock': AttentionBlock
    })
attentions = [layer.output for layer in model.layers if 'attention' in layer.name]
attentions.append(model.output)
embedder = tf.keras.models.Model(inputs=model.input, outputs=attentions)
embedder.summary()
# %%
ds = iter(memory_bank_ds)
listing = open(cfg['TESTING_LIST'])
listing = [patch.split('\t') for patch in listing]
i = 0

# %%
x, y = next(ds)
x = tf.cast(x, tf.float32)
LUT, colour_patches = build_seg_cmap(components=cfg['CLASS_COMPONENTS'])
probs, logits = model(x, training=False)
x_Seg = tf.math.argmax(probs, axis=-1)
out = tf.nn.embedding_lookup(LUT[1:], x_Seg)
plt.figure(figsize=(19.20, 10.80))
plt.imshow(tf.concat([x[0], (x[0] + out[0]) / 2, out[0]], 1))
plt.grid(False)
plt.legend(handles=colour_patches)
plt.title(listing[i])
plt.tight_layout()
plt.show()
i += 1


# %%
def process_attention_map(attention_map):
    # Max pooling across the channels to collapse the attention map to 2D
    attention_map = tf.reduce_max(attention_map, axis=-1)
    # Normalize attention map for visualization
    attention_map -= tf.reduce_min(attention_map)
    attention_map /= tf.reduce_max(attention_map)
    return attention_map.numpy()


# %%
x, y = next(ds)
x = tf.cast(x, tf.float32)

# Generate the outputs from the embedder model
outputs = embedder(x, training=False)

# Remove the last item which is the final output of the model, not an attention output
attention_outputs = outputs[:-1]

# Check for empty attention outputs
if not attention_outputs:
    raise ValueError("No attention outputs to display.")

# Assuming x is a batch of images, and you want to plot the first one
input_image = x[0].numpy()  # Convert the first image of the batch to a numpy array

# Calculate subplot grid size
num_attention_layers = len(attention_outputs)
fig, axes = plt.subplots(1, num_attention_layers + 1, figsize=(3 * (num_attention_layers + 1), 5))
axes = axes.flatten()  # Flatten in case there is only one row of subplots

# Plot the input image
axes[0].imshow(input_image)
axes[0].set_title("Input Image")
axes[0].axis('off')

# Loop through each attention output and plot
for i, attention in enumerate(attention_outputs, start=1):
    attention_processed = process_attention_map(attention[0])  # Assume the first image's attention map
    ax = axes[i]
    im = ax.imshow(attention_processed, cmap='viridis', interpolation='nearest')
    ax.set_title(f"Attention Map {i}")
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03, )  # Add a colorbar for each map

plt.tight_layout()
plt.show()
