# import numpy as np
# import os
# import cv2
# from sklearn.model_selection import train_test_split

# # Load dataset
# def load_data(folder, img_size=(5, 5)):
#     X, Y = [], []
#     class_labels = os.listdir(folder)
#     for label_idx, label in enumerate(class_labels):
#         class_path = os.path.join(folder, label)
#         if os.path.isdir(class_path):
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 if img is not None:
#                     img = cv2.resize(img, img_size)
#                     X.append(img)
#                     Y.append(label_idx)
#     return np.array(X), np.array(Y)

# # Hàm tích chập
# def conv2d(X, kernel):
#     h, w = X.shape
#     kh, kw = kernel.shape
#     out_h, out_w = h - kh + 1, w - kw + 1
#     output = np.zeros((out_h, out_w))
    
#     for i in range(out_h):
#         for j in range(out_w):
#             output[i, j] = np.sum(X[i:i+kh, j:j+kw] * kernel)
    
#     return output

# # Hàm kích hoạt ReLU
# def relu(X):
#     return np.maximum(0, X)

# # Lớp fully connected
# def dense(X, weights, bias):
#     return np.dot(X, weights) + bias

# # Hàm mất mát MSE
# def mse_loss(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)

# # Gradient của MSE
# def mse_grad(y_true, y_pred):
#     return -2 * (y_true - y_pred) / y_true.size

# # Load dataset
# X, Y = load_data("C:/Personal/final_graduate/data")
# X = X / 255.0  # Chuẩn hóa dữ liệu về [0,1]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Khởi tạo tham số
# kernel = np.random.rand(3, 3)
# dense_weights = np.random.rand(9, 1)
# dense_bias = np.random.rand(1)

# # Learning rate
# lr = 0.01

# # Training
# for epoch in range(100):
#     total_loss = 0
#     correct = 0
    
#     for i in range(len(X_train)):
#         # Forward pass
#         conv_out = conv2d(X_train[i], kernel)
#         relu_out = relu(conv_out)
#         flat_out = relu_out.flatten()
#         dense_out = dense(flat_out, dense_weights, dense_bias)
#         pred_label = int(dense_out > 0.5)  # Giả sử nhị phân
        
#         # Accuracy
#         if pred_label == Y_train[i]:
#             correct += 1
        
#         # Tính loss
#         loss = mse_loss(Y_train[i], dense_out)
#         total_loss += loss
        
#         # Backpropagation
#         grad_loss = mse_grad(Y_train[i], dense_out)
#         grad_dense_w = np.outer(flat_out, grad_loss)
#         grad_dense_b = grad_loss
        
#         # Cập nhật tham số
#         dense_weights -= lr * grad_dense_w
#         dense_bias -= lr * grad_dense_b
    
#     accuracy = correct / len(X_train)
#     print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train)}, Accuracy: {accuracy}")



# import numpy as np
# import os
# import cv2
# from sklearn.model_selection import train_test_split

# # Load dataset
# def load_data(folder, img_size=(5, 5)):
#     X, Y = [], []
#     class_labels = os.listdir(folder)
#     for label_idx, label in enumerate(class_labels):
#         class_path = os.path.join(folder, label)
#         if os.path.isdir(class_path):
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 if img is not None:
#                     img = cv2.resize(img, img_size)
#                     X.append(img)
#                     Y.append(label_idx)
#     return np.array(X), np.array(Y)

# # Hàm tích chập
# def conv2d(X, kernel):
#     h, w = X.shape
#     kh, kw = kernel.shape
#     out_h, out_w = h - kh + 1, w - kw + 1
#     output = np.zeros((out_h, out_w))
    
#     for i in range(out_h):
#         for j in range(out_w):
#             output[i, j] = np.sum(X[i:i+kh, j:j+kw] * kernel)
    
#     return output

# # Hàm kích hoạt ReLU
# def relu(X):
#     return np.maximum(0, X)

# # Hàm Softmax
# def softmax(X):
#     exp_X = np.exp(X - np.max(X))  # Tránh overflow
#     return exp_X / np.sum(exp_X)

# # Lớp fully connected
# def dense(X, weights, bias):
#     return np.dot(X, weights) + bias

# # Hàm mất mát Cross-Entropy
# def cross_entropy_loss(y_true, y_pred):
#     return -np.log(y_pred[y_true])

# # Gradient của Cross-Entropy
# def cross_entropy_grad(y_true, y_pred):
#     grad = y_pred.copy()
#     grad[y_true] -= 1
#     return grad

# # Load dataset
# X, Y = load_data("C:/Personal/final_graduate/data")
# X = X / 255.0  # Chuẩn hóa dữ liệu về [0,1]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Khởi tạo tham số
# kernel1 = np.random.rand(3, 3)
# kernel2 = np.random.rand(3, 3)
# dummy_input = np.zeros((5, 5))
# conv1_out = conv2d(dummy_input, kernel1)
# relu1_out = relu(conv1_out)
# conv2_out = conv2d(relu1_out, kernel2)
# relu2_out = relu(conv2_out)
# flattened_size = relu2_out.size
# dense_weights = np.random.rand(flattened_size, len(set(Y)))  # Điều chỉnh kích thước phù hợp
# bias = np.random.rand(len(set(Y)))

# # Learning rate
# lr = 0.01
# batch_size = 16  # Mini-Batch

# # Training
# for epoch in range(100):
#     total_loss = 0
#     correct = 0
#     indices = np.random.permutation(len(X_train))
#     X_train, Y_train = X_train[indices], Y_train[indices]
    
#     for i in range(0, len(X_train), batch_size):
#         batch_X = X_train[i:i+batch_size]
#         batch_Y = Y_train[i:i+batch_size]
        
#         batch_loss = 0
#         grad_dense_w = np.zeros_like(dense_weights)
#         grad_dense_b = np.zeros_like(bias)
        
#         for j in range(len(batch_X)):
#             # Forward pass
#             conv1_out = conv2d(batch_X[j], kernel1)
#             relu1_out = relu(conv1_out)
#             conv2_out = conv2d(relu1_out, kernel2)
#             relu2_out = relu(conv2_out)
#             flat_out = relu2_out.flatten()
            
#             dense_out = dense(flat_out, dense_weights, bias)
#             prob = softmax(dense_out)
#             pred_label = np.argmax(prob)
            
#             # Accuracy
#             if pred_label == batch_Y[j]:
#                 correct += 1
            
#             # Tính loss
#             loss = cross_entropy_loss(batch_Y[j], prob)
#             batch_loss += loss
            
#             # Backpropagation
#             grad_loss = cross_entropy_grad(batch_Y[j], prob)
#             grad_dense_w += np.outer(flat_out, grad_loss)
#             grad_dense_b += grad_loss
        
#         # Cập nhật tham số (dùng trung bình batch)
#         dense_weights -= lr * (grad_dense_w / batch_size)
#         bias -= lr * (grad_dense_b / batch_size)
#         total_loss += batch_loss
    
#     accuracy = correct / len(X_train)
#     print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train)}, Accuracy: {accuracy}")



# import numpy as np
# import os
# import cv2
# from sklearn.model_selection import train_test_split

# # Load dataset
# def load_data(folder, img_size=(5, 5)):
#     X, Y = [], []
#     class_labels = os.listdir(folder)
#     for label_idx, label in enumerate(class_labels):
#         class_path = os.path.join(folder, label)
#         if os.path.isdir(class_path):
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 if img is not None:
#                     img = cv2.resize(img, img_size)
#                     X.append(img)
#                     Y.append(label_idx)
#     return np.array(X), np.array(Y)

# # Lớp Convolution
# class Conv2D:
#     def __init__(self, kernel_size):
#         self.kernel = np.random.randn(kernel_size, kernel_size)
#         self.grad = np.zeros_like(self.kernel)
    
#     def forward(self, X):
#         self.input = X  # Lưu input để backward
#         h, w = X.shape
#         kh, kw = self.kernel.shape
#         out_h, out_w = h - kh + 1, w - kw + 1
#         output = np.zeros((out_h, out_w))
        
#         for i in range(out_h):
#             for j in range(out_w):
#                 output[i, j] = np.sum(X[i:i+kh, j:j+kw] * self.kernel)
        
#         return output
    
#     def backward(self, grad_output, lr=0.01):
#         kh, kw = self.kernel.shape
#         self.grad.fill(0)
#         for i in range(grad_output.shape[0]):
#             for j in range(grad_output.shape[1]):
#                 self.grad += grad_output[i, j] * self.input[i:i+kh, j:j+kw]
        
#         self.kernel -= lr * self.grad
#         return np.ones_like(self.input)  # Trả về gradient để tiếp tục lan truyền

# # Lớp ReLU
# class ReLU:
#     def forward(self, X):
#         self.input = X
#         return np.maximum(0, X)
    
#     def backward(self, grad_output):
#         return grad_output * (self.input > 0) if self.input is not None else grad_output

# # Lớp Flatten
# class Flatten:
#     def forward(self, X):
#         self.input_shape = X.shape
#         return X.flatten()
    
#     def backward(self, grad_output):
#         return grad_output.reshape(self.input_shape)

# # Lớp Fully Connected
# class Dense:
#     def __init__(self, input_size, output_size):
#         self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
#         self.bias = np.zeros(output_size)
    
#     def forward(self, X):
#         self.input = X
#         return np.dot(X, self.weights) + self.bias
    
#     def backward(self, grad_output, lr=0.01):
#         grad_weights = np.outer(self.input, grad_output)
#         grad_bias = grad_output
        
#         self.weights -= lr * grad_weights
#         self.bias -= lr * grad_bias
        
#         return np.dot(grad_output, self.weights.T)

# # Softmax & Cross-Entropy Loss
# def softmax(X):
#     exp_X = np.exp(X - np.max(X))
#     return exp_X / np.sum(exp_X)

# def cross_entropy_loss(y_true, y_pred):
#     y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
#     return -np.log(y_pred[y_true])

# def cross_entropy_grad(y_true, y_pred):
#     grad = y_pred.copy()
#     grad[y_true] -= 1
#     return grad

# # Load dataset
# X, Y = load_data("C:/Personal/final_graduate/data")
# X = X / 255.0
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Khởi tạo mô hình
# conv1 = Conv2D(3)
# relu1 = ReLU()
# conv2 = Conv2D(3)
# relu2 = ReLU()
# flatten = Flatten()

# # Xác định output size của Flatten
# sample_input = np.zeros((5, 5))  # Dựa vào img_size=(5,5)
# out = conv1.forward(sample_input)
# out = relu1.forward(out)
# out = conv2.forward(out)
# out = relu2.forward(out)
# out = flatten.forward(out)
# flatten_output_size = out.size  # Lấy kích thước đúng

# dense = Dense(flatten_output_size, len(set(Y)))

# # Training
# lr = 0.01
# batch_size = 16
# for epoch in range(100):
#     total_loss = 0
#     correct = 0
#     indices = np.random.permutation(len(X_train))
#     X_train, Y_train = X_train[indices], Y_train[indices]
    
#     for i in range(0, len(X_train), batch_size):
#         batch_X = X_train[i:i+batch_size]
#         batch_Y = Y_train[i:i+batch_size]
        
#         batch_loss = 0
#         for j in range(len(batch_X)):
#             # Forward pass
#             out = conv1.forward(batch_X[j])
#             out = relu1.forward(out)
#             out = conv2.forward(out)
#             out = relu2.forward(out)
#             out = flatten.forward(out)
#             out = dense.forward(out)
#             prob = softmax(out)
#             pred_label = np.argmax(prob)
            
#             # Accuracy
#             if pred_label == batch_Y[j]:
#                 correct += 1
            
#             # Loss
#             loss = cross_entropy_loss(batch_Y[j], prob)
#             batch_loss += loss
            
#             # Backpropagation
#             grad_loss = cross_entropy_grad(batch_Y[j], prob)
#             grad_out = dense.backward(grad_loss, lr)
#             grad_out = flatten.backward(grad_out)
#             grad_out = relu2.backward(grad_out)
#             grad_out = conv2.backward(grad_out, lr)
#             grad_out = relu1.backward(grad_out)
#             conv1.backward(grad_out, lr)
        
#         total_loss += batch_loss
    
#     accuracy = correct / len(X_train)
#     print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train)}, Accuracy: {accuracy}")


# import numpy as np
# import os
# import cv2
# from sklearn.model_selection import train_test_split

# # Định nghĩa kích thước ảnh động
# img_size = (32, 32)  # Có thể thay đổi thành bất kỳ kích thước nào mong muốn

# # Load dataset
# def load_data(folder, img_size):
#     X, Y = [], []
#     class_labels = os.listdir(folder)
#     for label_idx, label in enumerate(class_labels):
#         class_path = os.path.join(folder, label)
#         if os.path.isdir(class_path):
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
#                 img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                 if img is not None:
#                     img = cv2.resize(img, img_size)
#                     X.append(img)
#                     Y.append(label_idx)
#     return np.array(X), np.array(Y)

# # Hàm tích chập
# def conv2d(X, kernel):
#     h, w = X.shape
#     kh, kw = kernel.shape
#     out_h, out_w = h - kh + 1, w - kw + 1
#     output = np.zeros((out_h, out_w))
    
#     for i in range(out_h):
#         for j in range(out_w):
#             output[i, j] = np.sum(X[i:i+kh, j:j+kw] * kernel)
    
#     return output

# # Hàm kích hoạt ReLU
# def relu(X):
#     return np.maximum(0, X)

# # Hàm Softmax
# def softmax(X):
#     exp_X = np.exp(X - np.max(X))  # Tránh overflow
#     return exp_X / np.sum(exp_X)

# # Lớp fully connected
# def dense(X, weights, bias):
#     return np.dot(X, weights) + bias

# # Hàm mất mát Cross-Entropy
# def cross_entropy_loss(y_true, y_pred):
#     return -np.log(y_pred[y_true])

# # Gradient của Cross-Entropy
# def cross_entropy_grad(y_true, y_pred):
#     grad = y_pred.copy()
#     grad[y_true] -= 1
#     return grad

# # Load dataset
# X, Y = load_data("C:/Personal/final_graduate/data", img_size=img_size)
# X = X / 255.0  # Chuẩn hóa dữ liệu về [0,1]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Khởi tạo tham số
# kernel1 = np.random.rand(3, 3)
# kernel2 = np.random.rand(3, 3)

# dummy_input = np.zeros(img_size)  # Linh động theo kích thước ảnh
# dummy_conv1 = conv2d(dummy_input, kernel1)
# dummy_relu1 = relu(dummy_conv1)
# dummy_conv2 = conv2d(dummy_relu1, kernel2)
# dummy_relu2 = relu(dummy_conv2)
# flattened_size = dummy_relu2.size  # Tự động tính kích thước Flatten

# dense_weights = np.random.rand(flattened_size, len(set(Y)))
# bias = np.random.rand(len(set(Y)))

# # Learning rate
# lr = 0.01
# batch_size = 16  # Mini-Batch

# # Training
# for epoch in range(100):
#     total_loss = 0
#     correct = 0
#     indices = np.random.permutation(len(X_train))
#     X_train, Y_train = X_train[indices], Y_train[indices]
    
#     for i in range(0, len(X_train), batch_size):
#         batch_X = X_train[i:i+batch_size]
#         batch_Y = Y_train[i:i+batch_size]
        
#         batch_loss = 0
#         grad_dense_w = np.zeros_like(dense_weights)
#         grad_dense_b = np.zeros_like(bias)
        
#         for j in range(len(batch_X)):
#             # Forward pass
#             conv1_out = conv2d(batch_X[j], kernel1)
#             relu1_out = relu(conv1_out)
#             conv2_out = conv2d(relu1_out, kernel2)
#             relu2_out = relu(conv2_out)
#             flat_out = relu2_out.flatten()
            
#             dense_out = dense(flat_out, dense_weights, bias)
#             prob = softmax(dense_out)
#             pred_label = np.argmax(prob)
            
#             # Accuracy
#             if pred_label == batch_Y[j]:
#                 correct += 1
            
#             # Tính loss
#             loss = cross_entropy_loss(batch_Y[j], prob)
#             batch_loss += loss
            
#             # Backpropagation
#             grad_loss = cross_entropy_grad(batch_Y[j], prob)
#             grad_dense_w += np.outer(flat_out, grad_loss)
#             grad_dense_b += grad_loss
        
#         # Cập nhật tham số (dùng trung bình batch)
#         dense_weights -= lr * (grad_dense_w / batch_size)
#         bias -= lr * (grad_dense_b / batch_size)
#         total_loss += batch_loss
    
#     accuracy = correct / len(X_train)
#     print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train)}, Accuracy: {accuracy}")

import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Load dataset
def load_data(folder, img_size=(28, 28)):
    X, Y = [], []
    class_labels = sorted(os.listdir(folder))  # Sắp xếp để đảm bảo nhãn nhất quán
    for label_idx, label in enumerate(class_labels):
        class_path = os.path.join(folder, label)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    X.append(img)
                    Y.append(label_idx)
    return np.array(X), np.array(Y)

# Hàm tích chập có padding
def conv2d(X, kernel, padding=1):
    X_padded = np.pad(X, pad_width=padding, mode='constant', constant_values=0)
    h, w = X.shape
    kh, kw = kernel.shape
    out_h, out_w = h - kh + 2 * padding + 1, w - kw + 2 * padding + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(X_padded[i:i+kh, j:j+kw] * kernel)
    
    return output

# Hàm Batch Normalization
def batch_norm(X):
    mean = np.mean(X)
    std = np.std(X) + 1e-8
    return (X - mean) / std

# Hàm kích hoạt ReLU
def relu(X):
    return np.maximum(0, X)

# Hàm Dropout
def dropout(X, drop_rate=0.2):
    mask = np.random.binomial(1, 1 - drop_rate, size=X.shape)
    return X * mask

# Hàm Softmax
def softmax(X):
    exp_X = np.exp(X - np.max(X))  # Tránh overflow
    return exp_X / np.sum(exp_X)

# Lớp Fully Connected
def dense(X, weights, bias):
    return np.dot(X, weights) + bias

# Hàm mất mát Cross-Entropy
def cross_entropy_loss(y_true, y_pred):
    return -np.log(y_pred[y_true] + 1e-8)  # Tránh log(0)

# Gradient của Cross-Entropy
def cross_entropy_grad(y_true, y_pred):
    grad = y_pred.copy()
    grad[y_true] -= 1
    return grad

# Load dataset
X, Y = load_data("C:/Personal/final_graduate/data", img_size=(28, 28))
X = X / 255.0  # Chuẩn hóa dữ liệu về [0,1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Khởi tạo tham số (Xavier Initialization)
def xavier_init(size):
    return np.random.randn(*size) * np.sqrt(1. / size[0])

kernel1 = xavier_init((3, 3))
kernel2 = xavier_init((3, 3))
dummy_input = np.zeros((28, 28))
conv1_out = conv2d(dummy_input, kernel1)
relu1_out = relu(batch_norm(conv1_out))
conv2_out = conv2d(relu1_out, kernel2)
relu2_out = relu(batch_norm(conv2_out))
flattened_size = relu2_out.size
dense_weights = xavier_init((flattened_size, len(set(Y))))
bias = np.zeros(len(set(Y)))

# Learning rate và Batch Size
lr = 0.001
batch_size = 32

# Training
for epoch in range(100):
    total_loss = 0
    correct = 0
    indices = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[indices], Y_train[indices]
    
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_Y = Y_train[i:i+batch_size]
        
        batch_loss = 0
        grad_dense_w = np.zeros_like(dense_weights)
        grad_dense_b = np.zeros_like(bias)
        
        for j in range(len(batch_X)):
            conv1_out = conv2d(batch_X[j], kernel1)
            relu1_out = relu(batch_norm(conv1_out))
            conv2_out = conv2d(relu1_out, kernel2)
            relu2_out = relu(batch_norm(conv2_out))
            flat_out = dropout(relu2_out.flatten(), drop_rate=0.2)
            
            dense_out = dense(flat_out, dense_weights, bias)
            prob = softmax(dense_out)
            pred_label = np.argmax(prob)
            
            if pred_label == batch_Y[j]:
                correct += 1
            
            loss = cross_entropy_loss(batch_Y[j], prob)
            batch_loss += loss
            
            grad_loss = cross_entropy_grad(batch_Y[j], prob)
            grad_dense_w += np.outer(flat_out, grad_loss)
            grad_dense_b += grad_loss
        
        dense_weights -= lr * (grad_dense_w / batch_size)
        bias -= lr * (grad_dense_b / batch_size)
        total_loss += batch_loss
    
    accuracy = correct / len(X_train)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train):.4f}, Accuracy: {accuracy:.4f}")

