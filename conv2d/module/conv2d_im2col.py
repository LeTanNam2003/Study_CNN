import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, lr=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = lr  

        # Xavier Initialization
        limit = np.sqrt(6 / (in_channels + out_channels))
        self.kernels = np.random.uniform(-limit, limit, 
                                         (out_channels, in_channels, kernel_size, kernel_size))
        print(f"[INIT] Kernels shape: {self.kernels.shape}")  # Debug kernel shape

    def im2col(self, input, kernel_size, stride=1):
        batch_size, in_channels, H, W = input.shape
        k = kernel_size
        H_out = (H - k) // stride + 1
        W_out = (W - k) // stride + 1

        cols = np.zeros((batch_size, H_out, W_out, in_channels, k, k))
        
        for y in range(H_out):
            for x in range(W_out):
                cols[:, y, x, :, :, :] = input[:, :, y * stride:y * stride + k, x * stride:x * stride + k]

        return cols.reshape(batch_size * H_out * W_out, in_channels * k * k)

    def forward(self, input_tensor):
        print(f"[FORWARD] Input shape before conversion: {input_tensor.shape}")

        # Nếu input là PyTorch tensor, chuyển về NumPy
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().cpu().numpy()
            print("[FORWARD] Converted input tensor from PyTorch to NumPy.")

        batch_size, in_channels, height, width = input_tensor.shape

        if self.padding > 0:
            input_tensor = np.pad(input_tensor, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        # Bước 1: Chuyển input thành dạng im2col
        input_col = self.im2col(input_tensor, self.kernel_size, self.stride)  # (batch_size * H_out * W_out, in_channels * k * k)
        kernels_col = self.kernels.reshape(self.out_channels, -1)  # (out_channels, in_channels * k * k)

        # Bước 2: Tính output bằng phép nhân ma trận
        output_col = input_col @ kernels_col.T  # (batch_size * H_out * W_out, out_channels)
        output = output_col.reshape(batch_size, output_height, output_width, self.out_channels)
        output = output.transpose(0, 3, 1, 2)  # Đưa về dạng (batch_size, out_channels, H_out, W_out)

        # Chuyển output về Tensor PyTorch
        output_tensor = torch.tensor(output, dtype=torch.float32)
        print(f"[FORWARD] Output shape after conversion to PyTorch: {output_tensor.shape}")

        self.input = input_tensor  # Lưu input để dùng trong backward
        return output_tensor

    def backward(self, d_output):
        print(f"[BACKWARD] d_output shape: {d_output.shape}")
        batch_size, in_channels, height, width = self.input.shape
        out_channels, _, kernel_size, _ = self.kernels.shape

        # Bước 1: Chuyển input thành dạng im2col
        input_col = self.im2col(self.input, kernel_size, self.stride)  # (batch_size * H_out * W_out, in_channels * k * k)
        #d_output_col = d_output.transpose(0, 2, 3, 1).reshape(-1, out_channels)  # (batch_size * H_out * W_out, out_channels)
        d_output_col = d_output.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, out_channels)

        # Bước 2: Tính d_kernels bằng phép nhân ma trận
        d_kernels = d_output_col.T @ input_col  # (out_channels, in_channels * k * k)
        d_kernels = d_kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)

        # Bước 3: Cập nhật trọng số
        self.kernels -= self.lr * d_kernels
        self.d_kernels = d_kernels  # Lưu lại gradient để debug

        print("[BACKWARD] Backpropagation completed.")
        return d_kernels



