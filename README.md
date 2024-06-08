# Kaggle Dataset Downloader and Model Trainer

This repository provides a script to download VIBeID datasets, create DataLoaders for training and testing, and train a ResNet-18 and ResNet-50 model using PyTorch.

## Requirements
- Python 3.x
- `pip` (Python package installer)
- Kaggle API key (`kaggle.json`)

## Arguments

### `--kaggle_json` (required)
- **Description**: Path to the `kaggle.json` file for Kaggle API authentication.
- **Type**: `str`
- **Example**: `--kaggle_json kaggle.json`
- **Note**: This file is necessary for downloading datasets from Kaggle. It is available with the repo

### `--kaggle_dataset`
- **Description**: Kaggle dataset identifier in the format `mainakml/dataset-name`.
- **Type**: `str`
- **Default**: `'mainakml/vibeid-a-4-1'`
- **Example**: `--kaggle_dataset yourusername/yourdataset`
- **Note**: This argument specifies which dataset to download from Kaggle.

### `--output_dir`
- **Description**: Directory to download and unzip the Kaggle dataset.
- **Type**: `str`
- **Default**: `'vibeid-a-4-1/VIBeID_A_4_1'`
- **Example**: `--output_dir /path/to/output_dir`
- **Note**: The script will create this directory if it does not exist and will store the downloaded dataset here.

### `--batch_size`
- **Description**: Batch size for the DataLoader.
- **Type**: `int`
- **Default**: `16`
- **Example**: `--batch_size 16`
- **Note**: This determines the number of samples that will be propagated through the network at once.

### `--num_workers`
- **Description**: Number of worker threads to use for loading the data.
- **Type**: `int`
- **Default**: `2`
- **Example**: `--num_workers 4`
- **Note**: This is used to speed up data loading by using multiple threads.

### `--num_epochs`
- **Description**: Number of epochs to train the model.
- **Type**: `int`
- **Default**: `50`
- **Example**: `--num_epochs 30`
- **Note**: One epoch means that each sample in the dataset has had an opportunity to update the internal model parameters once.

### `--model`
- **Description**: Model type to use for training.
- **Type**: `str`
- **Choices**: `['resnet18', 'resnet50']`
- **Default**: `resnet18`
- **Example**: `--model resnet50`
- **Note**: Specifies which ResNet model architecture to use.

### `--num_classes`
- **Description**: Number of output classes for the model.
- **Type**: `int`
- **Default**: `15`
- **Example**: `--num_classes 20`
- **Note**: This should match the number of classes in your dataset.

## Step-by-Step guide
### STEP 1: Install Libraries:
python install_libraries.py

### STEP 1: Download the Datasets:
1. vibeid-a1 <a href="https://www.kaggle.com/datasets/mainakml/vibeid-a1">
2. vibeid-a2 <a href="https://www.kaggle.com/datasets/mainakml/vibeid-a2">
3. vibeid-a3<a href="https://www.kaggle.com/datasets/mainakml/vibeid-a3">
4. vibeid-a4 <a href="https://www.kaggle.com/datasets/mainakml/vibeid-a-4-1">

### STEP 2: Run
### single_image_run
python single_image_run.py --kaggle_json kaggle.json --kaggle_dataset mainakml/vibeid-a-4-1 --batch_size 16 --num_workers 2 --num_epochs 50
### multi_image_run
python multi_image_run.py --kaggle_json kaggle.json --kaggle_dataset mainakml/vibeid-a-4-1 --batch_size 16 --num_workers 2 --num_epochs 50
