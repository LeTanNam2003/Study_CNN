import torch
import torch.nn as nn

# Define batch size = 2, single channel (grayscale)
batch_size = 2
in_channels = 1
height = 5
width = 5

# Create two 5x5 images in a batch
input_tensor = torch.tensor([
    [[
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25]
    ]],
    [[
        [2, 4, 6, 8, 10],
        [12, 14, 16, 18, 20],
        [22, 24, 26, 28, 30],
        [32, 34, 36, 38, 40],
        [42, 44, 46, 48, 50]
    ]]
], dtype=torch.float32)  # Shape: (2, 1, 5, 5)

# Define a 3x3 kernel (same for all batch images)
kernel = torch.tensor([
    [[
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ]]
], dtype=torch.float32)  # Shape: (1, 1, 3, 3)

# Create a Conv2d layer with no bias
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)

# Set the weight manually
conv.weight.data = kernel

# Perform convolution
output = conv(input_tensor)

# Print output
print("Output after convolution:")
print(output.squeeze())  # Remove extra dimensions for readability

