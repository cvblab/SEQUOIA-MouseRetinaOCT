import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import OCTDataset  # Custom dataset class for OCT images
from model import UNet
import torch.nn.functional as F

# ============================ #
#       Configuration          #
# ============================ #

# Define generic paths (replace with actual paths when running)
base_folder = "/path/to/validation/volumes/"
model_path = "/path/to/trained/model/best_model.pth"
img_width, img_height = 256, 512  # Image dimensions

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_coef_ponderate(y_true, y_pred, epsilon=1e-6):
    """
    Computes the weighted Dice coefficient for multi-class segmentation.

    This function assumes `y_true` is a one-hot encoded tensor and `y_pred`
    is either logits or softmax probabilities.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (N, C, H, W).
        y_pred (torch.Tensor): Predicted tensor of shape (N, C, H, W).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Dice coefficient averaged across all classes.
    """
    y_true = y_true.float()
    y_pred = y_pred.float()
    intersection = 2 * torch.sum(y_true * y_pred, dim=(0, 2, 3))
    union = torch.sum(y_true * y_true, dim=(0, 2, 3)) + torch.sum(y_pred * y_pred, dim=(0, 2, 3))
    dice = (intersection + epsilon) / (union + epsilon)
    return torch.mean(dice)


# ============================ #
#         Load Model           #
# ============================ #

# Load the trained U-Net model
model = UNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# ============================ #
#       Evaluation Loop        #
# ============================ #

# Dictionary to store per-volume Dice scores
dice_per_volume = {}

# Iterate through each volume folder
for volume_name in os.listdir(base_folder):
    vol_path = os.path.join(base_folder, volume_name)
    if not os.path.isdir(vol_path):
        continue

    print(f"🔍 Processing volume: {volume_name}")

    # Load volume-specific dataset and dataloader
    dataset = OCTDataset(vol_path, img_width, img_height)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    dice_scores = []

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)            # Shape: (1, 1, H, W)
            masks = masks.to(device)          # Shape: (1, C, H, W)

            outputs = model(imgs)             # Shape: (1, C, H, W)
            preds = F.softmax(outputs, dim=1) # Predicted probabilities

            # Compute Dice score
            dice = dice_coef_ponderate(masks, outputs)
            dice_scores.append(dice.item())

    # Store scores for this volume
    dice_per_volume[volume_name] = dice_scores
    print(f"✅ {volume_name}: Mean Dice = {np.mean(dice_scores):.4f}")


# ============================ #
#         Summary Stats        #
# ============================ #

# Flatten all dice scores
all_dices = [d for scores in dice_per_volume.values() for d in scores]

# Print global statistics
print(f"\n📊 Global Dice Score: Mean = {np.mean(all_dices):.4f}, Std = {np.std(all_dices):.4f}")

# Detailed stats per volume
for vol, scores in dice_per_volume.items():
    print(f"📁 {vol}: Mean = {np.mean(scores):.4f}, Std = {np.std(scores):.4f}")

