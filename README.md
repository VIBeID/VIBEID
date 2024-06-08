# **VIBeID**: A Structural **VIB**ration-based Soft Biometric Dataset and Benchmark for Person **ID**entification
This repository provides a script to download Pre-processed  VIBeID datasets, create DataLoaders for training and testing, and train a ResNet-18 and ResNet-50 model using PyTorch.

![1717854965704](image/README/1717854965704.png)
## Requirements
- Python 3.x
- `pip` (Python package installer)
- Kaggle API key (`kaggle.json`) [optional]

## Arguments

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
- **Example**: `--num_classes 15/30/40/100`
- **Note**: This should match the number of classes in your dataset.

## Step-by-Step guide

### Quick Run 
- Run Multi-class Classification (Single Image) - ![single_run_demo.ipynb](https://github.com/Mainak1792/VIBEID/blob/main/single_run_demo.ipynb)
- Run Multi-class Classification (Multi Image)- ![multi_run_demo.ipynb](https://github.com/Mainak1792/VIBEID/blob/main/multi_run_demo.ipynb)
- Run Domain Adaptation demo - ![domain_adaptation_demo.ipynb](https://github.com/Mainak1792/VIBEID/blob/main/domain_adaptation_demo.ipynb)
### STEP 1: Install Libraries:
python install_libraries.py


### STEP 1: Download the Datasets:
You can download the datasets from the Kaggle (dataset is public)

1. vibeid-a1 [A1](https://www.kaggle.com/datasets/mainakml/vibeid-a1)
2. vibeid-a2 [A2](https://www.kaggle.com/datasets/mainakml/vibeid-a2)
3. vibeid-a3 [A3](https://www.kaggle.com/datasets/mainakml/vibeid-a3)
4. vibeid-a4 [A4](https://www.kaggle.com/datasets/mainakml/vibeid-a-4-1)

OR 
run 

```python kaggle_dataset_download.py --kaggle_dataset "mainakml/dataset link"```

Quick  Run 
```python kaggle_dataset_download.py --kaggle_dataset "mainakml/vibeid-a-4-1"```

change the dataset link as your requirement
1. mainakml/vibeid-a1
2. mainakml/vibeid-a2
3. mainakml/vibeid-a3
4. mainakml/vibeid-a-4-1


### STEP 2: Quick Run

```python single_run.py --output_dir C:\Users\mainak\Documents\GitHub\VIBEID\VIBeID_A_4_1 --batch_size 16 --num_epochs 100 --model resnet18 --num_classes 15```

### STEP 3: Run dataset as per your requirement

### single_image_run
```python single_run.py --output_dir "add dataset link which contains train and test" --batch_size 16 --num_epochs 100 --model resnet18 --num_classes 15/30/40/100```


### multi_image_run
```python multi_run.py --output_dir "add dataset link which contains train and test" --batch_size 16 --num_epochs 100 --model resnet18 --num_classes 15/30/40/100```


---

