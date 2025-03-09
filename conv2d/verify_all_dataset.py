import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os
from module import Conv2D  # Import Conv2D tự viết

# =============================
# 1️⃣ Load Dataset từ thư mục
# =============================
dataset_path = "C:/Personal/final_graduate/data/meningioma/"  # Cập nhật đường dẫn đúng
image_paths = glob.glob(os.path.join(dataset_path, "*.jpg"))

# =============================
# 2️⃣ Hàm xử lý ảnh đầu vào
# =============================
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  
    image = image.astype(np.float32) / 255.0  
    return np.expand_dims(np.expand_dims(image, axis=0), axis=0)  # (1, 1, H, W)

# =============================
# 3️⃣ Khởi tạo Mô hình Conv2D
# =============================
class TorchConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(x)

torch_conv = TorchConv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
torch_conv.conv.weight.data = torch.tensor(np.random.randn(1, 1, 3, 3), dtype=torch.float32, requires_grad=True)

conv_custom = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_custom.kernels = torch_conv.conv.weight.detach().numpy()

# =============================
# 4️⃣ Chạy trên toàn bộ dataset
# =============================
torch_times, custom_times = [], []

for image_path in image_paths:
    input_np = preprocess_image(image_path)
    input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

    # PyTorch Conv2D
    start_time = time.perf_counter()
    torch_output = torch_conv(input_torch)
    torch_time = time.perf_counter() - start_time
    torch_times.append(torch_time)

    # Custom Conv2D
    start_time = time.perf_counter()
    custom_output = conv_custom.forward(input_np)
    custom_time = time.perf_counter() - start_time
    custom_times.append(custom_time)

# =============================
# 5️⃣ Kết quả tổng hợp
# =============================
print(f"🔥 PyTorch Conv2D Avg Time: {np.mean(torch_times):.6f} sec")
print(f"🔥 Custom Conv2D Avg Time: {np.mean(custom_times):.6f} sec")

# =============================
# 6️⃣ Hiển thị 3 ảnh mẫu
# =============================
sample_images = np.random.choice(image_paths, 3, replace=False)

plt.figure(figsize=(12, 4))

for idx, image_path in enumerate(sample_images):
    input_np = preprocess_image(image_path)
    input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

    torch_output = torch_conv(input_torch).detach().numpy()
    custom_output = conv_custom.forward(input_np)

    plt.subplot(3, 3, idx * 3 + 1)
    plt.imshow(input_np[0, 0], cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(3, 3, idx * 3 + 2)
    plt.imshow(torch_output[0, 0], cmap="gray")
    plt.title("PyTorch Conv2D Output")
    plt.axis("off")

    plt.subplot(3, 3, idx * 3 + 3)
    plt.imshow(custom_output[0, 0], cmap="gray")
    plt.title("Custom Conv2D Output")
    plt.axis("off")

plt.show()
