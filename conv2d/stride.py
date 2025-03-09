import torch
import torch.nn as nn
import numpy as np
from module import Conv2D  # Import your custom Conv2D class

# === Step 1: Define Input Matrix (5x5) ===
input_matrix = torch.tensor([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=torch.float32)

# Reshape for Conv2D format (Batch, Channels, Height, Width)
input_tensor = input_matrix.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 5, 5)

# === Step 2: Define Kernel (3x3) ===
kernel = torch.tensor([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)

# === Step 3: Apply Custom Conv2D ===
custom_conv = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
custom_conv.kernels = kernel.numpy()  # Set the kernel manually
custom_output = custom_conv.forward(input_tensor.numpy())  # Get output

# === Step 4: Apply nn.Conv2d (for Verification) ===
torch_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
torch_conv.weight.data = kernel  # Set the same kernel
torch_output = torch_conv(input_tensor)  # Get output

# === Step 5: Print & Compare Results ===
print("Custom Conv2D Output:")
print(custom_output.squeeze())  # Remove extra dimensions

print("\nPyTorch Conv2D Output:")
print(torch_output.squeeze().detach().numpy())  # Convert to NumPy for easy comparison

# === Step 6: Compute Difference ===
max_diff = np.abs(custom_output - torch_output.detach().numpy()).max()
mean_diff = np.abs(custom_output - torch_output.detach().numpy()).mean()
print(f"\n[INFO] Max Difference: {max_diff}")
print(f"[INFO] Mean Difference: {mean_diff}")

# If max_diff and mean_diff are close to 0, your padding implementation is correct!

