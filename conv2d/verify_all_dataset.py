# import time
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import glob
# import os
# from module import Conv2D  # Import Conv2D tự viết
# import sys
# import time

# # Mở file log và ghi toàn bộ stdout vào file log
# log_file = open("log1.txt", "w")
# sys.stdout = log_file
# # =============================
# # 1️⃣ Load Dataset từ thư mục
# # =============================
# dataset_path = "C:/Personal/final_graduate/data/meningioma/"  # Cập nhật đường dẫn đúng
# image_paths = glob.glob(os.path.join(dataset_path, "*.jpg"))

# # =============================
# # 2️⃣ Hàm xử lý ảnh đầu vào
# # =============================
# def preprocess_image(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.resize(image, (128, 128))  
#     image = image.astype(np.float32) / 255.0  
#     return np.expand_dims(np.expand_dims(image, axis=0), axis=0)  # (1, 1, H, W)

# # =============================
# # 3️⃣ Khởi tạo Mô hình Conv2D
# # =============================
# class TorchConv2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

#     def forward(self, x):
#         return self.conv(x)

# torch_conv = TorchConv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
# torch_conv.conv.weight.data = torch.tensor(np.random.randn(1, 1, 3, 3), dtype=torch.float32, requires_grad=True)

# conv_custom = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
# conv_custom.kernels = torch_conv.conv.weight.detach().numpy()

# # =============================
# # 4️⃣ Chạy trên toàn bộ dataset
# # =============================
# torch_times, custom_times = [], []

# for image_path in image_paths:
#     input_np = preprocess_image(image_path)
#     input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

#     # PyTorch Conv2D
#     start_time = time.perf_counter()
#     torch_output = torch_conv(input_torch)
#     torch_time = time.perf_counter() - start_time
#     torch_times.append(torch_time)

#     # Custom Conv2D
#     start_time = time.perf_counter()
#     custom_output = conv_custom.forward(input_np)
#     custom_time = time.perf_counter() - start_time
#     custom_times.append(custom_time)

# # =============================
# # 5️⃣ Kết quả tổng hợp
# # =============================
# print(f"PyTorch Conv2D Avg Time: {np.mean(torch_times):.6f} sec")
# print(f"Custom Conv2D Avg Time: {np.mean(custom_times):.6f} sec")

# # =============================
# # 6️⃣ Hiển thị 3 ảnh mẫu
# # =============================
# sample_images = np.random.choice(image_paths, 3, replace=False)

# plt.figure(figsize=(12, 4))

# for idx, image_path in enumerate(sample_images):
#     input_np = preprocess_image(image_path)
#     input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

#     torch_output = torch_conv(input_torch).detach().numpy()
#     custom_output = conv_custom.forward(input_np)

#     plt.subplot(3, 3, idx * 3 + 1)
#     plt.imshow(input_np[0, 0], cmap="gray")
#     plt.title("Original Image")
#     plt.axis("off")

#     plt.subplot(3, 3, idx * 3 + 2)
#     plt.imshow(torch_output[0, 0], cmap="gray")
#     plt.title("PyTorch Conv2D Output")
#     plt.axis("off")

#     plt.subplot(3, 3, idx * 3 + 3)
#     plt.imshow(custom_output[0, 0], cmap="gray")
#     plt.title("Custom Conv2D Output")
#     plt.axis("off")
# log_file.close()

# plt.show()
import time
import torch
import torch.nn as nn
import numpy as np
import glob
import os
import cv2

# Import Conv2D tự viết
from module import Conv2D  

# =============================
# 1️⃣ Load Dataset
# =============================
dataset_path = "C:/Personal/final_graduate/data/meningioma/"  # Cập nhật đường dẫn đúng
image_paths = glob.glob(os.path.join(dataset_path, "*.jpg"))

if not image_paths:
    raise FileNotFoundError(f"Không tìm thấy ảnh trong thư mục: {dataset_path}")

# =============================
# 2️⃣ Hàm xử lý ảnh đầu vào
# =============================
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Lỗi đọc ảnh: {image_path}")

    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0  
    return np.expand_dims(np.expand_dims(image, axis=0), axis=0)  # (1, 1, H, W)

# =============================
# 3️⃣ Khởi tạo Mô hình
# =============================
torch_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
torch.nn.init.xavier_uniform_(torch_conv.weight)  # Khởi tạo trọng số tốt hơn

conv_custom = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_custom.kernels = torch_conv.weight.detach().cpu().numpy()  # Đồng bộ kernel

# =============================
# 4️⃣ Kiểm tra trên toàn dataset
# =============================
total_diff_input = []
total_diff_weight = []

for idx, image_path in enumerate(image_paths):
    print(f"[INFO] Đang xử lý ảnh {idx+1}/{len(image_paths)}: {image_path}")

    input_np = preprocess_image(image_path)
    input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

    # Forward
    torch_output = torch_conv(input_torch)
    custom_output = conv_custom.forward(input_np)

    # Kiểm tra kích thước đầu ra
    if torch_output.shape != custom_output.shape:
        print(f"[WARNING] Output shape mismatch! PyTorch: {torch_output.shape}, Custom: {custom_output.shape}")

    # Backward
    loss_torch = torch_output.sum()
    loss_torch.backward()  # Tính gradient cho PyTorch

    grad_torch_input = input_torch.grad.cpu().numpy()
    grad_torch_weight = torch_conv.weight.grad.cpu().numpy()

    # Custom Conv2D backward
    grad_custom_input, grad_custom_weight = conv_custom.backward(np.ones_like(custom_output))

    # Tính độ chênh lệch
    diff_input = np.abs(grad_torch_input - grad_custom_input).mean()
    diff_weight = np.abs(grad_torch_weight - grad_custom_weight).mean()

    total_diff_input.append(diff_input)
    total_diff_weight.append(diff_weight)

# =============================
# 5️⃣ Tổng hợp kết quả
# =============================
avg_diff_input = np.mean(total_diff_input)
avg_diff_weight = np.mean(total_diff_weight)

print(f"Gradient Input Avg Diff: {avg_diff_input:.6f}")
print(f"Gradient Weight Avg Diff: {avg_diff_weight:.6f}")

