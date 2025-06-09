"""
Training pipeline for U-Net model on OCT segmentation data.

This script loads preprocessed OCT images, applies reproducibility settings,
configures data loaders, defines the U-Net model, and trains it with or without validation.

Note: When training without validation, the number of epochs is fixed to the
best epoch previously observed during training with validation. This ensures
the model used for pseudo-mask inference has optimal performance.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset import OCTDataset
from model import UNet
from train import train_model

# ---------------------------
# Reproducibility (Seed Setup)
# ---------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# Configuration Parameters
# ---------------------------
data_folder = "<data_folder>"       
output_folder = "<output_folder>"   

batch_size = 8
num_epochs = 300                    # Default number of epochs (used with validation)
learning_rate = 0.001
img_width, img_height = 256, 512
use_validation = False               # Set to True to train with all available data

os.makedirs(output_folder, exist_ok=True)

# ---------------------------
# Device Configuration
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA devices:", torch.cuda.device_count())
    print("Current device ID:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

# ---------------------------
# Dataset Loading
# ---------------------------
train_dataset = OCTDataset(os.path.join(data_folder, "training"), img_width, img_height)
val_dataset = OCTDataset(os.path.join(data_folder, "validation"), img_width, img_height)

# ---------------------------
# Training Mode: With Validation
# ---------------------------
if use_validation:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet().to(device)

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        output_folder=output_folder,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

# ---------------------------
# Training Mode: Without Validation (Full Dataset)
# ---------------------------
else:
    # The number of epochs is set to the best epoch previously obtained
    # during training with validation. This allows training on the full dataset
    # to maximize data usage while avoiding overfitting, and provides the best
    # model for pseudo-mask inference.

    # Set to the best epoch previously observed in validation.
    # This ensures the model trained on the full dataset mirrors the most
    # performant configuration for pseudo-mask inference.
    num_epochs = <best_validation_epoch>  # Example: 109

    full_dataset = ConcatDataset([train_dataset, val_dataset])
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)

    train_model(
        train_loader=full_loader,
        val_loader=None,
        model=model,
        output_folder=output_folder,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )