import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# Tích chập với im2col
def im2col(X, kernel_size, stride=1, padding=1):
    X_padded = np.pad(X, ((padding, padding), (padding, padding)), mode='constant')
    kh, kw = kernel_size
    out_h = (X.shape[0] - kh + 2 * padding) // stride + 1
    out_w = (X.shape[1] - kw + 2 * padding) // stride + 1
    cols = np.zeros((kh * kw, out_h * out_w))

    for i in range(out_h):
        for j in range(out_w):
            patch = X_padded[i:i+kh, j:j+kw].flatten()
            cols[:, i * out_w + j] = patch

    return cols

def conv2d(X, kernel, stride=1, padding=1):
    kh, kw = kernel.shape
    cols = im2col(X, (kh, kw), stride, padding)
    kernel_flat = kernel.flatten()
    return (kernel_flat @ cols).reshape((X.shape[0] - kh + 2 * padding) // stride + 1, 
                                        (X.shape[1] - kw + 2 * padding) // stride + 1)

# Batch Normalization (vector hóa hoàn toàn)
def batch_norm(X, eps=1e-8):
    mean, std = np.mean(X), np.std(X)
    return (X - mean) / (std + eps)

# Các hàm khác
def relu(X):
    return np.maximum(0, X)

def dropout(X, drop_rate=0.2):
    mask = np.random.binomial(1, 1 - drop_rate, size=X.shape) / (1 - drop_rate)
    return X * mask

def softmax(X):
    exp_X = np.exp(X - np.max(X))
    return exp_X / np.sum(exp_X)

def dense(X, weights, bias):
    return np.dot(X, weights) + bias

def cross_entropy_loss(y_true, y_pred):
    return -np.log(y_pred[y_true] + 1e-8)

def cross_entropy_grad(y_true, y_pred):
    grad = y_pred.copy()
    grad[y_true] -= 1
    return grad

# Load dataset
def load_data(folder, img_size=(28, 28)):
    X, Y = [], []
    class_labels = sorted(os.listdir(folder))
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

# Chia tập dữ liệu thành train (70%), validation (15%) và test (15%)
X, Y = load_data("C:/Personal/final_graduate/data", img_size=(28, 28))
X = X / 255.0  # Chuẩn hóa dữ liệu về [0,1]

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

print(f"Train set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")

# Khởi tạo tham số (Xavier Initialization)
def xavier_init(size):
    return np.random.randn(*size) * np.sqrt(1. / size[0])

kernel1 = xavier_init((3, 3))
kernel2 = xavier_init((3, 3))
dense_weights = xavier_init((28 * 28, len(set(Y))))
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

        batch_conv1 = np.array([conv2d(img, kernel1) for img in batch_X])
        batch_relu1 = relu(batch_norm(batch_conv1))
        batch_conv2 = np.array([conv2d(img, kernel2) for img in batch_relu1])
        batch_relu2 = relu(batch_norm(batch_conv2))
        batch_flat = np.array([dropout(img.flatten(), drop_rate=0.2) for img in batch_relu2])

        dense_out = np.dot(batch_flat, dense_weights) + bias
        probs = np.apply_along_axis(softmax, 1, dense_out)

        pred_labels = np.argmax(probs, axis=1)
        correct += np.sum(pred_labels == batch_Y)

        losses = np.array([cross_entropy_loss(y, p) for y, p in zip(batch_Y, probs)])
        total_loss += np.sum(losses)

        grads = np.array([cross_entropy_grad(y, p) for y, p in zip(batch_Y, probs)])
        grad_dense_w = np.dot(batch_flat.T, grads) / batch_size
        grad_dense_b = np.mean(grads, axis=0)

        dense_weights -= lr * grad_dense_w
        bias -= lr * grad_dense_b

    accuracy = correct / len(X_train)
    
    # Validation Loss & Accuracy
    val_loss = 0
    val_correct = 0
    for i in range(len(X_val)):
        conv1_out = conv2d(X_val[i], kernel1)
        relu1_out = relu(batch_norm(conv1_out))
        conv2_out = conv2d(relu1_out, kernel2)
        relu2_out = relu(batch_norm(conv2_out))
        flat_out = relu2_out.flatten()

        dense_out = dense(flat_out, dense_weights, bias)
        prob = softmax(dense_out)
        pred_label = np.argmax(prob)

        val_loss += cross_entropy_loss(Y_val[i], prob)
        if pred_label == Y_val[i]:
            val_correct += 1

    val_loss /= len(X_val)
    val_accuracy = val_correct / len(X_val)

    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(X_train):.4f}, Train Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
