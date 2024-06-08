import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

required_packages = [
    "argparse",
    "os",
    "kaggle",
    "opencv-python",
    "subprocess",
    "torch",
    "torch.nn",
    "torch.optim",
    "torch.utils.data",
    "torchvision"
]

for package in required_packages:
    install_and_import(package)

# Additional imports for submodules
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
print("libraries installed ")