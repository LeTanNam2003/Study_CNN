# import time
# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from module import Conv2D  # Import Conv2D tự viết

# # =============================
# # 1️⃣ Load Image & Preprocess
# # =============================
# image_path = "C:/Personal/final_graduate/data/meningioma/Tr-me_0031.jpg"  
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (128, 128))  
# image = image.astype(np.float32) / 255.0  

# input_np = np.expand_dims(np.expand_dims(image, axis=0), axis=0)  # (1, 1, H, W)
# input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

# # =============================
# # 2️⃣ Define Models
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
# # 3️⃣ Training Loop
# # =============================
# epochs = 1  # Số epoch muốn train
# learning_rate = 0.001

# for epoch in range(epochs):
#     print(f"\n🚀 Epoch {epoch+1}/{epochs}")
    
#     # --------------------
#     # 🔹 Forward Pass
#     # --------------------
#     print(f"[CHECK] Input shape (Torch): {input_torch.shape}")
    
#     start_time = time.perf_counter()
#     torch_output = torch_conv(input_torch)
#     torch_time = time.perf_counter() - start_time
    
#     print(f"[CHECK] Output shape (Torch): {torch_output.shape}")

#     start_time = time.perf_counter()
#     custom_output = conv_custom.forward(input_np)
#     custom_time = time.perf_counter() - start_time

#     print(f"⏳ PyTorch Conv2D Time: {torch_time:.6f} sec | Custom Conv2D Time: {custom_time:.6f} sec")

#     # --------------------
#     # 🔹 Compute Loss
#     # --------------------
#     y_true = np.random.randn(*custom_output.shape)  
#     y_true_torch = torch.tensor(y_true, dtype=torch.float32, device=torch_output.device)  

#     loss_torch = torch.mean((torch_output - y_true_torch) ** 2)  
#     loss_torch.backward()

#     torch_d_kernels = torch_conv.conv.weight.grad.numpy()

#     d_output = 2 * (custom_output - y_true) / np.prod(custom_output.shape)
#     _, custom_d_kernels = conv_custom.backward(d_output)  

#     # --------------------
#     # 🔹 Update Kernels
#     # --------------------
#     torch_conv.conv.weight.data -= learning_rate * torch_conv.conv.weight.grad
#     conv_custom.kernels -= learning_rate * custom_d_kernels

#     # Reset gradient in PyTorch (để tránh tích lũy)
#     torch_conv.conv.weight.grad.zero_()

#     # --------------------
#     # 🔹 Compare Gradients
#     # --------------------
#     from numpy.linalg import norm

#     def cosine_similarity(a, b):
#         dot_product = np.dot(a.flatten(), b.flatten())
#         norm_a = norm(a.flatten())
#         norm_b = norm(b.flatten())
#         return dot_product / (norm_a * norm_b + 1e-8)

#     mse_error = np.mean((custom_d_kernels - torch_d_kernels) ** 2)
#     print(f"\n🔍 Comparing Gradients:")
#     print(f"🔹 MSE Error of Kernels Gradients: {mse_error:.10f}")
#     print(f"🔹 Cosine Similarity of Kernels Gradients: {cosine_similarity(custom_d_kernels, torch_d_kernels):.6f}")
    
#     # Check norm to debug floating point precision issues
#     print(f"Norm of custom gradient: {norm(custom_d_kernels.flatten()):.10f}")
#     print(f"Norm of PyTorch gradient: {norm(torch_d_kernels.flatten()):.10f}")

#     # --------------------
#     # 🔹 Check NaN Issues
#     # --------------------
#     if np.isnan(custom_d_kernels).any():
#         print("[ERROR] NaN detected in custom gradient!")
#     if np.isnan(torch_d_kernels).any():
#         print("[ERROR] NaN detected in PyTorch gradient!")

# # =============================
# # 4️⃣ Display Final Results
# # =============================
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 3, 1)
# plt.imshow(image, cmap="gray")
# plt.title("Original Image")
# plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.imshow(torch_output.detach().numpy()[0, 0], cmap="gray")
# plt.title("PyTorch Conv2D Output")
# plt.axis("off")

# plt.subplot(1, 3, 3)
# plt.imshow(custom_output[0, 0], cmap="gray")
# plt.title("Custom Conv2D Output")
# plt.axis("off")

# plt.show()
import numpy as np

def im2col(input, kernel_size, stride=1, padding=0):
    batch_size, in_channels, H, W = input.shape
    k = kernel_size

    # Thêm padding
    input_padded = np.pad(input, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1

    cols = np.zeros((batch_size, H_out, W_out, in_channels, k, k))

    for y in range(H_out):
        for x in range(W_out):
            cols[:, y, x, :, :, :] = input_padded[:, :, y * stride:y * stride + k, x * stride:x * stride + k]

    return cols.reshape(batch_size, -1)

def col2im(cols, input_shape, kernel_size, stride=1, padding=0):
    batch_size, in_channels, H, W = input_shape
    k = kernel_size

    H_out = (H + 2 * padding - k) // stride + 1
    W_out = (W + 2 * padding - k) // stride + 1

    cols_reshaped = cols.reshape(batch_size, H_out, W_out, in_channels, k, k)
    d_input = np.zeros((batch_size, in_channels, H + 2 * padding, W + 2 * padding))
    count_matrix = np.zeros((batch_size, in_channels, H + 2 * padding, W + 2 * padding))

    for y in range(H_out):
        for x in range(W_out):
            d_input[:, :, y * stride:y * stride + k, x * stride:x * stride + k] += cols_reshaped[:, y, x, :, :, :]
            count_matrix[:, :, y * stride:y * stride + k, x * stride:x * stride + k] += 1

    count_matrix[count_matrix == 0] = 1
    d_input /= count_matrix  # Chuẩn hóa cộng dồn

    if padding > 0:
        d_input = d_input[:, :, padding:-padding, padding:-padding]

    return d_input

# Test chương trình
np.random.seed(42)
input_shape = (1, 1, 130, 130)
kernel_size = 3
stride = 1
padding = 0

# Tạo dữ liệu ngẫu nhiên
input_data = np.random.randn(*input_shape)

# Thực hiện im2col
cols = im2col(input_data, kernel_size, stride, padding)

# Thực hiện col2im
output_data = col2im(cols, input_shape, kernel_size, stride, padding)

# Kiểm tra độ chênh lệch giữa input và output
error = np.abs(input_data - output_data).max()
print(f"[TEST] Max Error: {error:.6f}")

# Kiểm tra vùng biên (cần bằng 0)
extra_region = output_data[:, :, 128:, 128:]
print(f"[CHECK COL2IM] Extra region (128:, 128:): min={extra_region.min():.6f}, max={extra_region.max():.6f}, mean={extra_region.mean():.6f}")

# Nếu sai lệch quá lớn, cần debug lại
assert error < 1e-5, "❌ col2im có lỗi! Output khác input!"
assert np.all(extra_region == 0), "❌ col2im có lỗi, phần biên không bằng 0!"
print("✅ col2im hoạt động đúng!")

