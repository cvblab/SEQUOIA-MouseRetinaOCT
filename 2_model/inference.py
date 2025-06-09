import os
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader, Dataset
from model import UNet
import cv2

# ============================ #
#     Generic Directory Setup  #
# ============================ #

# Define generic paths (replace with actual paths when deploying)
input_folder = "/path/to/input/mat_files/"
output_folder = "/path/to/output/results/"
model_path = "/path/to/trained/model/best_model.pth"

# Create output directories if they don't exist
os.makedirs(output_folder, exist_ok=True)


# ============================ #
#        Load Trained Model    #
# ============================ #

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained U-Net model
model = UNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


class OCTDataset(Dataset):
    """
    Dataset class for handling 3D OCT image volumes stored as NumPy arrays.

    Each volume has shape (H, W, D) where D is the number of slices (frames).
    The dataset reshapes the volume into a list of 2D images.

    Args:
        frames (np.ndarray): A 3D NumPy array of shape (H, W, D) containing OCT slices.

    Attributes:
        frames (np.ndarray): Reordered volume to shape (D, H, W) for indexing.
    """

    def __init__(self, frames):
        self.frames = np.moveaxis(frames, -1, 0)  # Convert shape from (H, W, D) to (D, H, W)

    def __len__(self):
        """
        Returns:
            int: Number of slices (images) in the volume.
        """
        return self.frames.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses a single image slice.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape (1, H, W), normalized to [0, 1].
        """
        img = self.frames[idx]
        img = np.expand_dims(img, axis=0)  # Shape: (1, H, W)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img, dtype=torch.float32)
        return img


# ============================ #
#      Inference on Volumes    #
# ============================ #

# Process each .mat file in the input folder
for file in os.listdir(input_folder):
    if file.endswith(".mat"):
        mat_path = os.path.join(input_folder, file)
        data = sio.loadmat(mat_path)

        if 'resized_images' in data:
            frames = data['resized_images']  # 3D image volume: shape (H, W, D)
            try:
                frames = data["images"]
            except KeyError:
                try:
                    img = data["resized_images"]
                except KeyError:
                    print(f"Neither 'images' nor 'resized_images' found in {mat_path}")
                    continue
            dataset = OCTDataset(frames)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            masks = []
            with torch.no_grad():
                for idx, img in enumerate(dataloader):
                    img = img.to(device)
                    output = model(img)  # Output shape: (1, 2, H, W)
                    output = torch.sigmoid(output)  # Normalize logits to [0, 1]
                    mask = output.argmax(dim=1).cpu().numpy().astype(np.float32)  # Convert to class map
                    masks.append(mask)

            # Reconstruct volume of predicted masks
            masks = np.array(masks).squeeze()  # Shape: (N, H, W)
            masks = np.moveaxis(masks, 0, -1)  # Convert to (H, W, N) for MATLAB compatibility

            # Save as .mat file
            save_path = os.path.join(output_folder, file)
            sio.savemat(save_path, {'masks': masks})
            print(f"Saved as float32: {save_path}")
        else:
            print(f"'resized_images' key not found in {file}")



