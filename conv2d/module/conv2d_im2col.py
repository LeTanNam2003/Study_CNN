# import numpy as np
# import torch

# class Conv2D:
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, lr=0.01):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.lr = lr  

#         # Xavier Initialization
#         limit = np.sqrt(6 / (in_channels + out_channels))
#         self.kernels = np.random.uniform(-limit, limit, 
#                                          (out_channels, in_channels, kernel_size, kernel_size))
#         print(f"[INIT] Kernels shape: {self.kernels.shape}")

#     def im2col(self, input, kernel_size, stride=1):
#         batch_size, in_channels, H, W = input.shape
#         k = kernel_size
#         H_out = (H - k) // stride + 1
#         W_out = (W - k) // stride + 1
        
#         print(f"[DEBUG] H_out: {H_out}, W_out: {W_out}, Input shape: {input.shape}")

#         cols = np.zeros((batch_size, H_out, W_out, in_channels, k, k))

#         for y in range(H_out):
#             for x in range(W_out):
#                 cols[:, y, x, :, :, :] = input[:, :, y * stride:y * stride + k, x * stride:x * stride + k]  
#         return cols.reshape(batch_size * H_out * W_out, in_channels * k * k)

#     def forward(self, input_tensor):
#         print(f"[FORWARD] Input shape before conversion: {input_tensor.shape}")

#         if isinstance(input_tensor, torch.Tensor):
#             input_tensor = input_tensor.detach().cpu().numpy()
#             print("[FORWARD] Converted input tensor from PyTorch to NumPy.")

#         batch_size, in_channels, height, width = input_tensor.shape

#         if self.padding > 0:
#             input_tensor = np.pad(input_tensor, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

#         output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
#         output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

#         input_col = self.im2col(input_tensor, self.kernel_size, self.stride)
#         kernels_col = self.kernels.reshape(self.out_channels, -1)
#         output_col = input_col @ kernels_col.T
#         output = output_col.reshape(batch_size, output_height, output_width, self.out_channels)
#         output = output.transpose(0, 3, 1, 2)

#         output_tensor = torch.tensor(output, dtype=torch.float32)
#         print(f"[FORWARD] Output shape after conversion to PyTorch: {output_tensor.shape}")

#         self.input = input_tensor
#         return output_tensor

#     def col2im(self, cols, input_shape, kernel_size, stride=1):
#         batch_size, in_channels, H, W = input_shape
#         H_out = (H - kernel_size) // stride + 1
#         W_out = (W - kernel_size) // stride + 1

#         d_input = np.zeros((batch_size, in_channels, H, W))
#         expected_shape = (batch_size, H_out, W_out, in_channels, kernel_size, kernel_size)
#         print(f"[DEBUG] cols.shape: {cols.shape}, expected_shape: {expected_shape}")
        
#         if cols.size != np.prod(expected_shape):
#             raise ValueError(f"[ERROR] Size mismatch: expected {np.prod(expected_shape)}, got {cols.size}")

#         cols_reshaped = cols.reshape(expected_shape)

#         for y in range(kernel_size):
#             for x in range(kernel_size):
#                 #d_input[:, :, y:y + H_out * stride:stride, x:x + W_out * stride:stride] += cols_reshaped[:, :, :, :, y, x]
#                 d_input[:, :, y:y + H_out * stride:stride, x:x + W_out * stride:stride] += cols_reshaped[:, :, :, :, y, x].transpose(0, 3, 1, 2)

#         return d_input

#     def backward(self, d_output):
#         print(f"[BACKWARD] d_output shape: {d_output.shape}")
#         batch_size, out_channels, output_height, output_width = d_output.shape
#         in_channels, kernel_size = self.in_channels, self.kernel_size

#         if isinstance(d_output, torch.Tensor):
#             d_output_np = d_output.detach().cpu().numpy()
#         else:
#             d_output_np = d_output

#         d_output_col = d_output_np.transpose(0, 2, 3, 1).reshape(-1, out_channels)
#         input_col = self.im2col(self.input, kernel_size, self.stride)  

#         d_kernels = d_output_col.T @ input_col
#         d_kernels = d_kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)

#         kernels_col = self.kernels.reshape(out_channels, -1)
#         d_input_col = d_output_col @ kernels_col
#         d_input = self.col2im(d_input_col, self.input.shape, kernel_size, self.stride)

#         self.kernels -= self.lr * d_kernels
#         self.d_kernels = d_kernels

#         print("[BACKWARD] Backpropagation completed.")
#         return d_input, d_kernels


# import numpy as np
# import torch

# class Conv2D:
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, lr=0.01):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.lr = lr  

#         # Xavier Initialization
#         limit = np.sqrt(6 / (in_channels + out_channels))
#         self.kernels = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
#         print(f"[INIT] Kernels shape: {self.kernels.shape}")  # Debug kernel shape

#     def im2col(self, input, kernel_size, stride=1):
#         batch_size, in_channels, H, W = input.shape
#         k = kernel_size
#         H_out = (H - k) // stride + 1
#         W_out = (W - k) // stride + 1
        

#         cols = np.zeros((batch_size, H_out, W_out, in_channels, k, k))
        
#         for y in range(H_out):
#             for x in range(W_out):
#                 cols[:, y, x, :, :, :] = input[:, :, y * stride:y * stride + k, x * stride:x * stride + k]

#         print(f"[DEBUG] im2col output shape: {cols.shape}")  # Kiểm tra kích thước output của im2col
#         return cols.reshape(batch_size * H_out * W_out, in_channels * k * k)

#     def col2im(cols, X_shape, K, stride, padding):
#         """Chuyển đổi từ dạng cột về ảnh gốc"""
#         N, C, H, W = X_shape
#         H_padded = H + 2 * padding
#         W_padded = W + 2 * padding
#         H_out = (H_padded - K) // stride + 1
#         W_out = (W_padded - K) // stride + 1
        
#         X_padded = np.zeros((N, C, H_padded, W_padded))
#         cols = cols.reshape(N, C, K, K, H_out, W_out)
        
#         for i in range(K):
#             for j in range(K):
#                 X_padded[:, :, i:i + H_out * stride:stride, j:j + W_out * stride:stride] += cols[:, :, i, j, :, :]
        
#         return X_padded[:, :, padding:H + padding, padding:W + padding]


#     def forward(dout, X, W, stride=1, padding=0):
#         """
#         dout: Gradient của đầu ra có shape (N, C_out, H_out, W_out)
#         X: Đầu vào gốc có shape (N, C_in, H_in, W_in)
#         W: Trọng số có shape (C_out, C_in, K, K)
#         stride: Bước nhảy
#         padding: Số pixel padding
#         """
#         N, C_in, H_in, W_in = X.shape
#         C_out, C_in, K, _ = W.shape
#         N, C_out, H_out, W_out = dout.shape
        
#         # Gradient của bias (db)
#         db = np.sum(dout, axis=(0, 2, 3))
        
#         # Chuyển đổi X và dout sang dạng cột
#         X_col = im2col(X, K, stride, padding)
#         dout_reshaped = dout.reshape(N, C_out, -1)
        
#         # Tính gradient của W
#         dW = np.einsum("nij,njk->ik", dout_reshaped, X_col)
#         dW = dW.reshape(C_out, C_in, K, K)
        
#         # Tính gradient của X
#         W_reshaped = W.reshape(C_out, -1)  # (C_out, C_in*K*K)
#         dX_col = np.einsum("ji,nij->njk", W_reshaped, dout_reshaped)
#         dX = col2im(dX_col, X.shape, K, stride, padding)
        
#         return dX, dW, db
   




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

        # Xavier Initialization
        limit = np.sqrt(6 / (in_channels + out_channels))
        self.kernels = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
        print(f"[INIT] Kernels shape: {self.kernels.shape}")

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
        print(f"[FORWARD] Input shape: {input_tensor.shape}")

        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.detach().cpu().numpy()

        batch_size, in_channels, height, width = input_tensor.shape

        if self.padding > 0:
            input_tensor = np.pad(input_tensor, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        input_col = self.im2col(input_tensor, self.kernel_size, self.stride)
        kernels_col = self.kernels.reshape(self.out_channels, -1)

        output_col = input_col @ kernels_col.T  
        output = output_col.reshape(batch_size, output_height, output_width, self.out_channels)
        output = output.transpose(0, 3, 1, 2)

        output_tensor = torch.tensor(output, dtype=torch.float32)
        print(f"[FORWARD] Output shape: {output_tensor.shape}, min: {output_tensor.min()}, max: {output_tensor.max()}")

        self.input = input_tensor
        return output_tensor

    def backward(self, d_output):
        print(f"[BACKWARD] d_output shape: {d_output.shape}")
        
        batch_size, in_channels, height, width = self.input.shape
        out_channels, _, kernel_size, _ = self.kernels.shape

        if isinstance(d_output, torch.Tensor):
            d_output = d_output.detach().cpu().numpy()

        d_output_col = d_output.transpose(0, 2, 3, 1).reshape(-1, out_channels)
        input_col = self.im2col(self.input, kernel_size, self.stride)

        d_kernels = d_output_col.T @ input_col  
        d_kernels = d_kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)
        
        print(f"[GRADIENT] Kernel Gradient shape: {d_kernels.shape}, min: {d_kernels.min()}, max: {d_kernels.max()}")
        
        d_input_col = d_output_col @ self.kernels.reshape(out_channels, -1)  
        d_input = self.col2im(d_input_col, self.input.shape, kernel_size, self.stride)

        if self.padding > 0:
            d_input = d_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        self.kernels -= self.lr * d_kernels  
        print(f"[UPDATE] Kernels updated, min: {self.kernels.min()}, max: {self.kernels.max()}")

        return d_input, d_kernels
    
    def col2im(self, cols, input_shape, kernel_size, stride=1):
        batch_size, in_channels, H, W = input_shape
        k = kernel_size
        H_out = (H - k) // stride + 1
        W_out = (W - k) // stride + 1  

        expected_size = batch_size * H_out * W_out * in_channels * k * k
        if cols.size != expected_size:
            raise ValueError(f"[ERROR] Mismatch in size: cols has {cols.size} elements, expected {expected_size}")

        cols_reshaped = cols.reshape(batch_size, H_out, W_out, in_channels, k, k)
        
        d_input = np.zeros((batch_size, in_channels, H, W))

        for y in range(H_out):
            for x in range(W_out):
                d_input[:, :, y * stride:y * stride + k, x * stride:x * stride + k] += cols_reshaped[:, y, x, :, :, :]

        print(f"[DEBUG] col2im output shape: {d_input.shape}")
        return d_input


