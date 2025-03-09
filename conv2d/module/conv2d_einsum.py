import numpy as np
import torch

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, lr=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = lr  

        limit = np.sqrt(6 / (in_channels + out_channels))
        self.kernels = np.random.uniform(-limit, limit, 
                                         (out_channels, in_channels, kernel_size, kernel_size))
        print(f"[INIT] Kernels shape: {self.kernels.shape}")

    def im2col(self, input, kernel_size, stride=1):
        batch_size, in_channels, H, W = input.shape
        k = kernel_size
        H_out = (H - k) // stride + 1
        W_out = (W - k) // stride + 1
        
        cols = np.zeros((batch_size, in_channels, k, k, H_out, W_out))
        for i in range(k):
            for j in range(k):
                cols[:, :, i, j, :, :] = input[:, :, i:i + H_out * stride:stride, j:j + W_out * stride:stride]
        
        return cols.reshape(batch_size * H_out * W_out, in_channels * k * k)

    def forward(self, input_tensor):
        print(f"[FORWARD] Input shape before conversion: {input_tensor.shape}")

        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().cpu().numpy()
            print("[FORWARD] Converted input tensor from PyTorch to NumPy.")

        batch_size, in_channels, height, width = input_tensor.shape

        if self.padding > 0:
            input_tensor = np.pad(input_tensor, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        input_col = self.im2col(input_tensor, self.kernel_size, self.stride)
        kernels_col = self.kernels.reshape(self.out_channels, -1)

        output_col = np.einsum('bohwi,oi->bohw', input_col.reshape(batch_size, output_height, output_width, in_channels * self.kernel_size * self.kernel_size), kernels_col)
        
        output_tensor = torch.tensor(output_col, dtype=torch.float32)
        print(f"[FORWARD] Output shape after conversion to PyTorch: {output_tensor.shape}")
        
        self.input = input_tensor  
        return output_tensor

    def backward(self, d_output):
        print(f"[BACKWARD] d_output shape: {d_output.shape}")
        batch_size, in_channels, height, width = self.input.shape
        out_channels, _, kernel_size, _ = self.kernels.shape

        input_col = self.im2col(self.input, kernel_size, self.stride)
        d_output_col = d_output.transpose(0, 2, 3, 1).reshape(-1, out_channels)

        d_kernels = np.einsum('bohw,bihw->oi', d_output_col.reshape(batch_size, height - kernel_size + 1, width - kernel_size + 1, out_channels), input_col.reshape(batch_size, height - kernel_size + 1, width - kernel_size + 1, in_channels * kernel_size * kernel_size))
        d_kernels = d_kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)

        self.kernels -= self.lr * d_kernels
        self.d_kernels = d_kernels  

        print("[BACKWARD] Backpropagation completed.")
        return d_kernels
