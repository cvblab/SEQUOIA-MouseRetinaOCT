import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import UNet

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
# Dice Coefficient (Weighted)
# ---------------------------
def dice_coef_ponderado(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Computes the weighted Dice coefficient across all classes.
    
    Args:
        y_true (torch.Tensor): Ground truth one-hot masks, shape (B, C, H, W).
        y_pred (torch.Tensor): Predicted softmax probabilities, shape (B, C, H, W).
        epsilon (float): Smoothing constant to avoid division by zero.

    Returns:
        torch.Tensor: Scalar tensor representing mean Dice coefficient across classes.
    """
    y_true = y_true.float()
    y_pred = y_pred.float()
    intersection = 2 * torch.sum(y_true * y_pred, dim=(0, 2, 3))
    union = torch.sum(y_true * y_true, dim=(0, 2, 3)) + torch.sum(y_pred * y_pred, dim=(0, 2, 3))
    dice = (intersection + epsilon) / (union + epsilon)
    return torch.mean(dice)

def dice_loss_ponderate(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the Dice loss (1 - Dice coefficient).

    Args:
        y_true (torch.Tensor): Ground truth one-hot masks.
        y_pred (torch.Tensor): Raw logits from the model (before softmax).

    Returns:
        torch.Tensor: Scalar tensor representing Dice loss.
    """
    y_pred = F.softmax(y_pred, dim=1)
    return 1 - dice_coef_ponderado(y_true, y_pred)

# ---------------------------
# Training Function
# ---------------------------
def train_model(train_loader, val_loader, model, output_folder, device, 
                num_epochs: int = 300, learning_rate: float = 0.001):
    """
    Trains the U-Net model using the given data loaders, optimizer, and Dice loss.
    
    Args:
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader or None): DataLoader for validation set (can be None for full training).
        model (nn.Module): Neural network model (U-Net).
        output_folder (str): Directory where checkpoints and logs will be saved.
        device (torch.device): Computation device ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.

    Outputs:
        - Saves the best model's state_dict as "best_model.pth" in `output_folder`.
        - Writes training and validation metrics to "best_model_info.txt".
    """
    
    criterion = dice_loss_ponderate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_val_dice = 0
    best_epoch = 0
    best_train_dice = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_dice = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(masks, outputs)
            preds = F.softmax(outputs, dim=1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice_coef_ponderado(masks, preds).item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        # ---------------------------
        # Validation Loop (if enabled)
        # ---------------------------
        if val_loader is not None:
            model.eval()
            val_dice = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    preds = F.softmax(outputs, dim=1)
                    val_dice += dice_coef_ponderado(masks, preds).item()
            val_dice /= len(val_loader)

            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")

            # Save best model based on validation Dice
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_train_dice = train_dice
                best_epoch = epoch + 1

                torch.save(model.state_dict(), os.path.join(output_folder, "best_model.pth"))
                with open(os.path.join(output_folder, "best_model_info.txt"), 'w') as f:
                    f.write(f"Best model saved at epoch: {best_epoch}\n")
                    f.write(f"Train Dice: {best_train_dice:.4f}\n")
                    f.write(f"Validation Dice: {best_val_dice:.4f}\n")
                print(f"New best model saved with Train Dice: {best_train_dice:.4f}, Val Dice: {best_val_dice:.4f}")

        # ---------------------------
        # No Validation Mode
        # ---------------------------
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")

            # Save final model after last epoch
            if epoch + 1 == num_epochs:
                torch.save(model.state_dict(), os.path.join(output_folder, "best_model.pth"))
                with open(os.path.join(output_folder, "best_model_info.txt"), 'w') as f:
                    f.write(f"Best model saved at epoch: {epoch+1}\n")
                    f.write(f"Train Dice: {train_dice:.4f}\n")
                print(f"Final model saved with Train Dice: {train_dice:.4f}")
