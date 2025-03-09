import numpy as np
import cv2
import matplotlib.pyplot as plt

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

    def forward(self, input_tensor):
        self.input = input_tensor  # Save input for backward pass
        batch_size, in_channels, height, width = input_tensor.shape

        if self.padding > 0:
            input_tensor = np.pad(input_tensor, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        output = np.zeros((batch_size, self.out_channels, output_height, output_width))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        region = input_tensor[b, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                        output[b, oc, i, j] = np.sum(region * self.kernels[oc, :, :, :])  

        return output

    def backward(self, d_output):
        batch_size, in_channels, height, width = self.input.shape
        out_channels, _, kernel_size, _ = self.kernels.shape

        # Gradient của kernel
        d_kernels = np.zeros_like(self.kernels)

        for oc in range(out_channels):
            for ic in range(in_channels):
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        h_range = min(self.input.shape[2] - i, d_output.shape[2])
                        w_range = min(self.input.shape[3] - j, d_output.shape[3])

                        d_kernels[oc, ic, i, j] = np.sum(
                            self.input[:, ic, i:i + h_range, j:j + w_range] * d_output[:, oc, :h_range, :w_range]
                        )

        # Gradient của input
        d_input = np.zeros_like(self.input)
        flipped_kernels = np.flip(self.kernels, axis=(2, 3))  

        pad_h = kernel_size - 1
        pad_w = kernel_size - 1
        d_output_padded = np.pad(d_output, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

        for b in range(batch_size):
            for ic in range(in_channels):
                for i in range(height):
                    for j in range(width):
                        region = d_output_padded[b, :, i:i + kernel_size, j:j + kernel_size]
                        d_input[b, ic, i, j] = np.sum(region * flipped_kernels[:, ic, :, :])

        # Cập nhật trọng số
        self.kernels -= self.lr * d_kernels
        self.d_kernels = d_kernels  # Lưu lại gradient để debug

        return d_input
