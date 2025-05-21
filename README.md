Below is an example of a detailed **README.md** file for your semantic segmentation project. This file explains the project’s purpose, required folder structure, installation instructions, and how to use the training and prediction scripts (including processing images and videos).

---

# Semantic Segmentation with SMP

This project provides a flexible framework for training and inference of semantic segmentation models using [segmentation_models_pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch). It supports multiple segmentation architectures such as:

- Unet
- Unet++ (UnetPlusPlus)
- FPN
- PSPNet
- DeepLabV3 (Can't work for now)
- DeepLabV3+(Can't work for now)
- Linknet
- MAnet
- PAN
- UPerNet
- Segformer

The framework is designed to be generic and can be applied to any semantic segmentation dataset with minimal changes.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Structure](#data-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction / Inference](#prediction--inference)
- [Project Structure](#project-structure)
- [Additional Notes](#additional-notes)

## Features

- **Multi-Model Support:** Easily switch between segmentation architectures via command-line parameters.
- **Pre-trained Backbones:** Utilize pre-trained encoder weights (e.g., from ImageNet) for faster convergence.
- **Flexible Data Loading:** Supports any semantic segmentation dataset provided that:
  - The input images are stored in an `images` folder.
  - The corresponding masks are stored in a `masks` folder.
  - The mask filenames must have the same base name as the image file, with the suffix `_mask.png` (e.g., `A.jpg` → `A_mask.png`).
  - A `_classes.csv` file is included that specifies the pixel value and the class name.
- **Video Processing:** The `predict.py` script can process both images and videos. Videos are converted to a common MP4 format using FFmpeg.
- **Easy-to-Use Training Pipeline:** The `train.py` script reads the dataset, builds the chosen model, and starts training while saving checkpoints periodically.

## Requirements

Make sure you have Python 3.7+ installed. The required Python packages are listed in `requirements.txt`.

## Installation

1. **Clone the repository:**

   ```bash
   git clone git@github.com:cby0730/CustomSemanticSegmentation.git
   cd CustomSemanticSegmentation
   ```

2. **Install dependencies:**

   It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt

   # Use pipenv virtual environment
   pipenv --python 3.10.14
   pipenv install -r requirements.txt
   ```

   The main dependencies include:
   - `torch` and `torchvision`
   - `segmentation-models-pytorch`
   - `tqdm`
   - `Pillow`
   - `numpy`
   - `opencv-python` (for video processing)
   - (FFmpeg must be installed on your system for video conversion)

## Data Structure

Organize your dataset as follows:

```
data/
├── train/
│   ├── images/           # Contains the training images (e.g., A.jpg, B.png, etc.)
│   ├── masks/            # Contains the corresponding segmentation masks.
│   │                      # IMPORTANT: Each mask file must have the same base name as its image
│   │                      # but with a "_mask.png" suffix (e.g., A_mask.png for image A.jpg).
│   └── _classes.csv      # CSV file defining the classes.
│                          # Example content:
│                           # Pixel Value,Class
│                           # 0,background
│                           # 1,road
│                           # 2,sidewalk
└── valid/
    ├── images/           # Validation images
    └── masks/            # Validation masks (same naming convention as above)
```

You can use any dataset that follows this structure. If your dataset has different classes, update the `_classes.csv` accordingly.

## Usage

### Training

To train a segmentation model, use the `train.py` script. You can select the model architecture and encoder backbone via command-line arguments.

#### Example command:

```bash
python train.py \
  --train_data_dir data/train \
  --valid_data_dir data/valid \
  --checkpoints_dir checkpoints \
  --epochs 50 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --model_name unet \
  --encoder_name resnet34 \
  --encoder_weights imagenet
```

#### Command-line Arguments Explanation:

- `--train_data_dir`: Path to the training data folder (should contain `images`, `masks`, and `_classes.csv`).
- `--valid_data_dir`: Path to the validation data folder (should contain `images` and `masks`).
- `--checkpoints_dir`: Directory where model checkpoints will be saved.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.
- `--learning_rate`: Learning rate for the optimizer.
- `--model_name`: Name of the segmentation model to use. Options include: `unet`, `unet++`, `fpn`, `pspnet`, `deeplabv3`, `deeplabv3+`, `linknet`, `manet`, `pan`, `upernet`, `segformer`.
- `--encoder_name`: Name of the encoder/backbone (e.g., `resnet34`, `mit_b0`, etc.).
- `--encoder_weights`: Pre-trained weights for the encoder (e.g., `imagenet`).

### Prediction / Inference

The `predict.py` script allows you to perform inference on both images and videos. It automatically detects file extensions and processes them accordingly.

#### Example command:

```bash
python predict.py \
  --model_path checkpoints/best_model.pth \
  --input_dir test \
  --output_dir predictions \
  --model_name segformer \
  --encoder_name mit_b0 \
  --encoder_weights imagenet
```

#### Command-line Arguments Explanation:

- `--model_path`: Path to the checkpoint file of the trained model (a `.pth` file).
- `--input_dir`: Directory containing input files for prediction (images and/or videos).
- `--output_dir`: Directory where the predicted segmentation masks and overlay images/videos will be saved.
- `--model_name`: The segmentation model to use (same options as for training).
- `--encoder_name`: The encoder/backbone for the model.
- `--encoder_weights`: Pre-trained weights used for the encoder.

#### Processing Details:

- **Images:** For each image, the script will:
  - Read the image.
  - Resize it to `640x640` (as set in the transform).
  - Run inference.
  - Save a segmentation result image with the suffix `_segmentation.png`.
  - Save an overlay image (original + segmentation) with the suffix `_overlay.png`.

- **Videos:** For each video file (e.g., `.mp4`, `.avi`, etc.), the script will:
  1. Convert the input video to MP4 format using FFmpeg.
  2. Process each frame to generate segmentation and overlay videos.
  3. Convert the output videos to a common MP4 format for compatibility.
  4. Save final segmentation and overlay videos with filenames ending in `_segmentation_final.mp4` and `_overlay_final.mp4` respectively.

The utility functions for video conversion are defined in `utils.py`.

## Project Structure After Training

```
your-project/
├── data/
│   ├── train/
│   │   ├── images/          # Training images
│   │   ├── masks/           # Corresponding masks (filenames: <name>_mask.png)
│   │   └── _classes.csv     # Class definitions
│   └── valid/
│       ├── images/          # Validation images
│       └── masks/           # Validation masks
├── checkpoints/             # Saved model checkpoints
├── dataset.py               # Data loading and transformation
├── model.py                 # Function to create segmentation models (supports multiple architectures)
├── trainer.py               # Training and validation pipeline
├── train.py                 # Script to launch training
├── predict.py               # Script for inference on images and videos
├── utils.py                 # Utility functions (e.g., video conversion with FFmpeg)
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```

## Project Structure After Inference

```
your-project/
├── data/
│   ├── train/
│   │   ├── images/          # Training images
│   │   ├── masks/           # Corresponding masks (filenames: <name>_mask.png)
│   │   └── _classes.csv     # Class definitions
│   └── valid/
│       ├── images/          # Validation images
│       └── masks/           # Validation masks
├── checkpoints/             # Saved model checkpoints
├── test/                    # Video or Image to test
├── predictions/             # Inferenced Video or Image
├── dataset.py               # Data loading and transformation
├── model.py                 # Function to create segmentation models (supports multiple architectures)
├── trainer.py               # Training and validation pipeline
├── train.py                 # Script to launch training
├── predict.py               # Script for inference on images and videos
├── utils.py                 # Utility functions (e.g., video conversion with FFmpeg)
├── requirements.txt         # Python package dependencies
└── README.md                # This file
```

## Additional Notes

- **FFmpeg:**  
  Make sure FFmpeg is installed on your system as it is used for video conversion. You can install it via your package manager (e.g., `sudo apt install ffmpeg` on Ubuntu) or download it from [FFmpeg.org](https://ffmpeg.org/).

- **Customization:**  
  You can modify the image transformation parameters (e.g., resize dimensions) in both `train.py` and `predict.py` if needed. Ensure that the input size is appropriate for your chosen model architecture (many require dimensions divisible by 32).

- **_classes.csv:**  
  The `_classes.csv` file should have two columns: `Pixel Value` and `Class`. This file is used to determine the number of classes in your dataset. If your dataset does not include a particular class (e.g., "objects"), make sure to update or remove it from the CSV accordingly.

- **Extensibility:**  
  The `get_model()` function in `model.py` is designed to easily add more architectures in the future. Simply follow the pattern shown to integrate additional models.

## Contact

For any questions or suggestions, please feel free to open an issue or contact the repository maintainer.

---

Happy segmenting!
```

---

This **README.md** should provide users with clear, step-by-step instructions on how to set up, train, and run inference (including processing both images and videos) with your semantic segmentation project. Adjust any paths or parameter defaults as needed for your specific environment.