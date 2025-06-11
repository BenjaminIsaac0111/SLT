# Attention U-Net for Medical Image Segmentation

## Overview
This project implements an Attention U-Net model for medical image segmentation tasks. The model is trained using TensorFlow and leverages mixed precision for improved performance on supported GPUs.

## Model Architecture

### Attention U-Net
The Attention U-Net model implemented in this project is designed for high-resolution medical image segmentation tasks. This model is an advanced version of the traditional U-Net, incorporating attention mechanisms and pixel shuffling for improved performance.

#### Features
- **Pixel Shuffling:** Used for up-sampling in the decoder part of the network, it helps in achieving a more refined up-sampling compared to traditional methods like transposed convolutions.
- **Attention Mechanisms:** Integrated at each level of the decoder to focus the model on relevant features for better context integration between the encoder and decoder pathways.
- **Group Normalization:** Utilised instead of Batch Normalization to stabilize the training process in smaller batch sizes of large medical images.

#### Encoder
The encoder progressively captures deeper features while reducing spatial dimensions using average pooling. Each convolutional block is followed by group normalization and incorporates residual connections to enhance feature propagation.

#### Decoder
Each step of the decoder includes up-sampling (either by pixel shuffling or traditional up-sampling), followed by an attention block that refines the feature map by focusing on relevant spatial areas. The decoder also uses skip connections from the encoder to recover spatial information lost during down-sampling.

#### Output
The final layer of the model is a convolutional layer that maps the features to the desired number of classes followed by a softmax layer for semantic segmentation tasks. The model can be configured to return logits or the softmax probabilities based on the requirement.

### Custom Layers
- **PixelShuffle:** Implements sub-pixel convolution, which sometimes helps in up-sampling the feature map more efficiently and with fewer artifacts compared to traditional methods.
- **AttentionBlock:** A custom implementation that computes the attention gating signal to modulate the feature response, enhancing the model's focus on important features during the decoding phase.

This architecture is designed to be robust yet flexible, allowing modifications to the number of layers, levels, and filters to adapt to different sizes and types of medical imaging datasets.

## Software Features
- Mixed precision support to enhance performance on modern GPUs.
- Distributed training using TensorFlow's `MirroredStrategy`.
- Custom data loading and augmentation pipeline.
- Implementation of custom losses and the use of class weights for imbalanced data.
- Configuration through YAML files for flexible experimentation.

## Prerequisites
Ensure you have the following installed:
- python==3.8+
- cudatoolkit==11.2
- cudnn==8.1.0
- tensorflow==2.10.1
- tensorflow-addons==0.19.0
- pyaml==23.5.8

You can create a Conda environment with these dependencies using the provided
`environment.yml` file:

```bash
conda env create -f environment.yml
```

This environment targets **Python 3.8** and installs TensorFlow **2.10.***.

### Data Preparations
The data pipeline is designed to handle image datasets for segmentation tasks in a convenient way. This section guides you through preparing and structuring your dataset to work seamlessly with the provided TensorFlow data loading and augmentation scripts.

#### Preparing the Dataset
1. **Organize Images**: Your dataset should consist of paired images where the left half is the input image and the right half contains the corresponding segmentation masks, stored as integer values representing various classes in the blue channel of the image.

2. **Split Data**: Split your dataset into training and testing subsets. Typically, a ratio of 80/20 or 70/30 is used for training to validation data.

3. **File Lists**: Create text files (For example, `train.txt` and `test.txt`) containing the filenames of the images in your training and testing datasets, respectively. Each line in these files should contain a single image file relative to the `DATA_DIR` configured in your settings (See Training Configuration section.).
   Example `train.txt` entry:

`relative/path/to/image1.png`

`relative/path/to/image2.png`

 or

`image1.png`

`image2.png`

## Training a Model

The Training pipeline is configured using a `.yaml` file with various parameters. Modify the configurations in configurations/configuration.yaml to suit your training needs and hardware setup. See cfg/example_config.yaml for an example configuration.

### Configuration
Ensure your configuration file (`configuration.yaml`) includes the correct paths and parameters relevant to your training run.

### General Settings
- **MODEL_DIR**: Path where model outputs, such as logs and saved models, will be stored. Default is `./Model_Example_Outputs/`.
- **MODEL_NAME**: The name for the saved model files. This will be used to save the final model and intermediate checkpoints. Example: `attention_resnet_example.h5`.

### Model Architecture
- **N_MODEL_LEVELS**: Specifies the number of levels in the U-Net architecture, where each level includes one down-sampling step in the encoder and one up-sampling step in the decoder. Example: `2`.
- **N_CONV_PER_LAYER**: Number of convolution blocks per level. Each block typically consists of a convolution followed by a normalization and activation function. Example: `2`.
- **USE_ATTENTION**: Determines whether additive attention blocks are included in the decoder for focusing on salient features. Set to `True` to enable.
- **USE_PIXEL_SHUFFLE**: Determine whether to use Pixel Shuffle for Upsampling. If False then default is Up-Sampling with bilinear interpolation. Set to `True` to enable.
- **N_FILTERS**: The number of filters in each convolution layer, constant across all layers, unlike the original U-Net which doubles the filters after each down-sampling. Example: `64`.
- **INPUT_SIZE**: Dimensions of the input images that the model will accept. Specified as height and width. Example: [1024, 512].
- **OUT_CHANNELS**: The number of output classes for segmentation. Each class corresponds to a different label in the segmentation task. Example: `9`.

### Training Configuration
- **LEARNING_RATE**: The initial learning rate for the optimizer. Example: `1.0e-05`.
- **USE_XLA**: Enables TensorFlow's XLA (Accelerated Linear Algebra) JIT compilation, which can improve performance by optimizing the model's computation graph.
- **USE_FOCAL_LOSS**: Use Focal Cross Entropy Loss. This can help with learning rarer examples in your dataset, else use a standard weighted cross entropy. Set to `True` to enable.
- **DATA_DIR**: Directory where training and testing data are located. Example: `Examples/`.
- **TRAINING_LIST**: File path containing a list of filenames for training data images. Example: `Examples/train.txt`.
- **TESTING_LIST**: File path containing a list of filenames for test data images used during validation. Example: `Examples/test.txt`.
- **SHUFFLE_BUFFER_SIZE**: The buffer size used by the shuffle operation in the dataset preparation. It helps in randomizing the input data during training. Example: `256`.
- **BATCH_SIZE**: Number of training examples utilized in one iteration. Example: `1`.
- **EPOCHS**: Total number of training cycles through the entire dataset. Example: `10`.
- **STEPS**: Total number of batches of samples to train in one epoch. Example: `237`.
- **VAL_BATCH_SIZE**: Number of examples per batch during validation. Typically, the same as `BATCH_SIZE` unless specified. Example: `1`.
- **VAL_STEPS**: Number of validation batches to execute at the end of each epoch. Example: `60`.

### Class Components and Weights
- **CLASS_COMPONENTS**: A dictionary that maps each class index to a meaningful name, helping to understand the model's outputs. Example mappings include:
  - `0`: Non-Informative
  - `1`: Tumour
  - `2`: Stroma, and so on.
- **CLASS_WEIGHTS**: List of weights for each class to handle class imbalance during training. Higher weights can be assigned to classes that are underrepresented or more important to detect accurately. 
This is applied to both the focal or cross entropy losses if set.

To train the model, run the following command with you specified configuration (see example file in cfg/):
```
main.py --config cfg/config_example.yaml
```
This script will train the model according to the parameters specified in the YAML configuration file. Training progress will be displayed on the console.


## Model Checkpointing
The training script automatically saves checkpoints and the best model based on validation loss.
Checkpoints are saved in the directory specified by the MODEL_DIR and MODEL_NAME configuration options.

## Validation and Testing
After training, you can validate the model performance on a separate test set configured in the YAML file. The validation step runs automatically at the end of each training epoch.

## Customization
You can customize the model architecture and training routine by modifying:

`Model/unets.py` and `Model/custom_layers.py`: For changes to the U-Net architecture.

`Losses/losses.py`: For custom loss functions.

`Processing/transforms.py`: For data augmentation and preprocessing methods.

## Contributing
Contributions to this project are welcome. Please ensure to follow the existing coding style and add unit tests for any new or changed functionality.

## License
Distributed under the GPL-3.0 License. See LICENSE for more information.

## Authors
Benjamin Isaac Wilson - benjamintaya0111@gmail.com
## Acknowledgments
This project was inspired by the U-Net architecture initially proposed for biomedical image segmentation and its Attention-UNet variation.
