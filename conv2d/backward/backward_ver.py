
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from module import Conv2D  # Import Conv2D tự viết

# =============================
# 1️⃣ Load Image & Preprocess
# =============================
image_path = "C:/Personal/final_graduate/data/meningioma/Tr-me_0031.jpg"  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 128))  
image = image.astype(np.float32) / 255.0  

input_np = np.expand_dims(np.expand_dims(image, axis=0), axis=0)  # (1, 1, H, W)
input_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)

# =============================
# 2️⃣ Monitor PyTorch Conv2D
# =============================
class TorchConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(x)

torch_conv = TorchConv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
torch_conv.conv.weight.data = torch.tensor(np.random.randn(1, 1, 3, 3), dtype=torch.float32, requires_grad=True)

start_time = time.perf_counter()
torch_output = torch_conv(input_torch)
torch_time = time.perf_counter() - start_time

# =============================
# 3️⃣ Monitor Custom Conv2D
# =============================
conv_custom = Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
conv_custom.kernels = torch_conv.conv.weight.detach().numpy()

start_time = time.perf_counter()
custom_output = conv_custom.forward(input_np)
custom_time = time.perf_counter() - start_time

# =============================
# 4️⃣ Print Results
# =============================
print(f"⏳ PyTorch Conv2D Time: {torch_time:.6f} sec")
print(f"⏳ Custom Conv2D Time: {custom_time:.6f} sec")

#=============================
# 4️⃣ Compute Gradients
# =============================
y_true = np.random.randn(*custom_output.shape)  
y_true_torch = torch.tensor(y_true, dtype=torch.float32, device=torch_output.device)  

loss_torch = torch.mean((torch_output - y_true_torch) ** 2)  
loss_torch.backward()

torch_d_kernels = torch_conv.conv.weight.grad.numpy()

d_output = 2 * (custom_output - y_true) / custom_output.size  
custom_d_kernels = conv_custom.backward(d_output)  

# =============================
# 5️⃣ Compare Gradients
# =============================
def compare_grads(name, grad1, grad2):
    denominator = np.abs(grad1) + np.abs(grad2) + 1e-8
    relative_error = np.abs(grad1 - grad2) / np.maximum(denominator, 1e-8)
    print(f"{name}: Max relative error: {np.max(relative_error):.6f}, Mean relative error: {np.mean(relative_error):.6f}")

print("\n🔍 Comparing Gradients:")
compare_grads("Kernel Gradients", conv_custom.d_kernels, torch_d_kernels)
# =============================
# 5️⃣ Display Results
# =============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(torch_output.detach().numpy()[0, 0], cmap="gray")
plt.title("PyTorch Conv2D Output")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(custom_output[0, 0], cmap="gray")
plt.title("Custom Conv2D Output")
plt.axis("off")

plt.show()
