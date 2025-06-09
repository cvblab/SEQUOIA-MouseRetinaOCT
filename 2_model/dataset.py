import os
import random
import numpy as np
import scipy.io
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------------------------
# Reproducibility
# ---------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# OCTDataset Class
# ---------------------------
class OCTDataset(Dataset):
    """
    Custom PyTorch Dataset for loading OCT (Optical Coherence Tomography) images and segmentation masks.

    This dataset assumes the following folder structure:
        folder_path/
        ├── Images/    # .mat files containing the raw images (key: 'images' or 'resized_images')
        └── Mask/      # .mat files containing binary segmentation masks (key: 'masks')

    The dataset loads and processes all samples at initialization, storing them in memory as PyTorch tensors.

    Args:
        folderimages (str): Path to the folder containing 'Images' and 'Mask' subdirectories.
        img_width (int): Desired width of the resized images and masks.
        img_height (int): Desired height of the resized images and masks.
    """

    def __init__(self, folderimages, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        self.image_files = []
        self.mask_files = []

        # Subfolders
        image_folder = os.path.join(folderimages, "Images")
        mask_folder = os.path.join(folderimages, "Mask")

        # Filter out non-.mat files and OS artifacts
        files = [
            f for f in os.listdir(image_folder)
            if f.endswith(".mat") and f not in [
                "Thumbs.db", "._.DS_S", "._.DS_Store", ".DS_Store"
            ]
        ]

        for file in files:
            self.image_files.append(os.path.join(image_folder, file))
            self.mask_files.append(os.path.join(mask_folder, file))

        # Preload all images and masks
        self.images, self.masks = self.load_and_preprocess()

    def load_and_preprocess(self):
        """
        Loads all image and mask pairs, resizes and normalizes them,
        and returns them as PyTorch tensors.

        Returns:
            images (torch.Tensor): Tensor of shape (B, 1, H, W)
            masks (torch.Tensor): Tensor of shape (B, 2, H, W) with one-hot encoding
        """
        images = []
        masks = []

        for img_path, mask_path in zip(self.image_files, self.mask_files):
            # Load image
            try:
                img = scipy.io.loadmat(img_path)["images"]
            except KeyError:
                try:
                    img = scipy.io.loadmat(img_path)["resized_images"]
                except KeyError:
                    print(f"Neither 'images' nor 'resized_images' found in {img_path}")
                    continue

            img = cv2.resize(img, (self.img_width, self.img_height))
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.float32) / 255.0
            img = np.clip(img, 0, 1)

            # Load mask
            mask = scipy.io.loadmat(mask_path)["masks"]
            mask = cv2.resize(mask, (self.img_width, self.img_height))
            mask = (mask > 0.5).astype(np.float32)  # Binarize

            images.append(img)
            masks.append(mask)

        # Convert to proper tensor shapes
        images = np.concatenate(images, axis=-1)  # Stack along depth
        images = np.moveaxis(images, -1, 0)       # (B, H, W)
        images = np.expand_dims(images, axis=1)   # (B, 1, H, W)

        masks = np.concatenate(masks, axis=-1)
        masks = np.moveaxis(masks, -1, 0)         # (B, H, W)

        # Convert to PyTorch tensors
        images = torch.tensor(images, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.long)

        # One-hot encode masks: (B, C, H, W)
        masks = F.one_hot(masks, num_classes=2).permute(0, 3, 1, 2).float()

        return images, masks

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves a sample by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (torch.Tensor): Image tensor of shape (1, H, W).
            mask (torch.Tensor): One-hot encoded mask tensor of shape (2, H, W).
        """
        return self.images[idx], self.masks[idx]
