import io
import sys

import os
import cv2
import numpy as np

import psutil
import math
import random
import copy
import time
import pickle
from scipy.signal import correlate, convolve
from matplotlib import image as mpimg
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import trange
from matplotlib import pyplot as plt

# @title helper functions

def plot_conv(convolutions=None, img_src='asharifiz.png', sequential=False):
    # Load the image
    img = mpimg.imread(img_src)
    cols = len(convolutions) + 1 if convolutions is not None else 1
    fig, axes = plt.subplots(1, cols, figsize=(15, 15 * cols))

    # Display the image
    axes[0].imshow(img)
    axes[0].axis('off')  # Hide the axis
    axes[0].set_title('Original')

    for i, conv in enumerate(convolutions):
        x = np.transpose(img, (2, 0, 1))
        x = np.expand_dims(x, axis=(0))
        out = conv(x).squeeze()
        out = np.transpose(out, (1, 2, 0))
        axes[i + 1].imshow(out)
        axes[i + 1].axis('off')  # Hide the axis
        axes[i + 1].set_title(f'Filter {i + 1}')
        if sequential:
            img = out

    plt.show()

def plot_results(losses, accuracies):
    # Plot the loss
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, kept_classes):
    dim = len(kept_classes)
    labels = [class_names[i] for i in kept_classes]
    # Plot the confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred)
    norm_conf_mat = conf_mat / np.sum(conf_mat, axis=1)
    # plot the matrix
    fig, ax = plt.subplots()
    plt.imshow(norm_conf_mat)
    plt.title('Confusion Matrix')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    plt.xticks(range(dim), labels, rotation=45)
    plt.yticks(range(dim), labels)
    plt.colorbar()
    # Put number of each cell in plot
    for i in range(dim):
        for j in range(dim):
            c = conf_mat[j, i]
            color = 'black' if c > 500 else 'white'
            ax.text(i, j, str(int(c)), va='center', ha='center', color=color)
    plt.show()


def get_data(filter_classes):
    fashion_mnist = fetch_openml("Fashion-MNIST", parser='auto')
    x, y = fashion_mnist['data'], fashion_mnist['target'].astype(int)
    # Remove classes
    filtered_indices = np.isin(y, filter_classes)
    x, y = x[filtered_indices].to_numpy(), y[filtered_indices]
    # Normalize the pixels to be in [-1, +1] range
    x = ((x / 255.) - .5) * 2
    removed_class_count = 0
    for i in range(10):  # Fix the labels
        if i in filter_classes and removed_class_count != 0:
            y[y == i] = i - removed_class_count
        elif i not in filter_classes:
            removed_class_count += 1
    # Do the train-test split
    return train_test_split(x, y, test_size=10_000)


def onehot_encoder(y, num_labels):
    one_hot = np.zeros(shape=(y.size, num_labels), dtype=int)
    one_hot[np.arange(y.size), y] = 1
    return one_hot


class Layer:
    def __init__(self):
        self.inp = None
        self.out = None

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return self.forward(inp)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, lr: float) -> None:
        pass

class Conv2D(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.w = 0.1 * np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.zeros((out_channels, 1))

    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp = inp
        batch_size, in_channels, height, width = inp.shape

        # Padding the input
        self.padded_inp = np.pad(inp, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.out = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Convolution operation
        for i in range(out_height):
            for j in range(out_width):
                region = self.padded_inp[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                self.out[:, :, i, j] = np.tensordot(region, self.w, axes=([1, 2, 3], [1, 2, 3])) + self.b.T

        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for Conv2D layer."""
        batch_size, in_channels, height, width = self.inp.shape
        _, _, out_height, out_width = up_grad.shape

        # Initialize gradients
        self.dw = np.zeros_like(self.w)
        self.db = np.sum(up_grad, axis=(0, 2, 3), keepdims=True).reshape(self.out_channels, 1)
        down_grad = np.zeros_like(self.padded_inp)

        # Gradient computation for weights and input
        for i in range(out_height):
            for j in range(out_width):
                region = self.padded_inp[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                self.dw += np.tensordot(up_grad[:, :, i, j], region, axes=([0], [0]))  # Compute weight gradient
                for n in range(batch_size):
                    down_grad[n, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += np.tensordot(self.w, up_grad[n, :, i, j], axes=(0, 0))

        # Remove padding if applied
        if self.padding > 0:
            down_grad = down_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return down_grad

    def step(self, lr: float) -> None:
        """Update weights and biases."""
        self.w -= lr * self.dw
        self.b -= lr * self.db

class MaxPool2D(Layer):
    def __init__(self, pool_size: int = 2, stride: int = 2):
        """Max Pooling Layer."""
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Forward pass of max pooling."""
        self.inp = inp
        batch_size, channels, height, width = inp.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        out = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                region = inp[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                out[:, :, i, j] = np.max(region, axis=(2, 3))

        self.out = out
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for max pooling."""
        batch_size, channels, height, width = self.inp.shape
        down_grad = np.zeros_like(self.inp)

        out_height, out_width = up_grad.shape[2], up_grad.shape[3]

        for i in range(out_height):
            for j in range(out_width):
                region = self.inp[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                max_mask = (region == np.max(region, axis=(2, 3), keepdims=True))
                down_grad[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size] += max_mask * up_grad[:, :, i, j][:, :, None, None]

        return down_grad
    
class AvgPool2D(Layer):
    def __init__(self, pool_size: int = 2, stride: int = 2):
        """Average Pooling Layer."""
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Forward pass of average pooling."""
        self.inp = inp
        batch_size, channels, height, width = inp.shape

        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        out = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                region = inp[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size]
                out[:, :, i, j] = np.mean(region, axis=(2, 3))

        self.out = out
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for average pooling."""
        batch_size, channels, height, width = self.inp.shape
        down_grad = np.zeros_like(self.inp)

        out_height, out_width = up_grad.shape[2], up_grad.shape[3]

        for i in range(out_height):
            for j in range(out_width):
                region_grad = up_grad[:, :, i, j][:, :, None, None] / (self.pool_size * self.pool_size)
                down_grad[:, :, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size] += region_grad

        return down_grad
class Flatten(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Flatten the input into a 2D array."""
        self.inp_shape = inp.shape
        return inp.reshape(self.inp_shape[0], -1)

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Reshape the gradient back to the original input shape."""
        return up_grad.reshape(self.inp_shape)

# @title NNs from scratch

class Layer:
    def __init__(self):
        self.inp = None
        self.out = None

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return self.forward(inp)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, lr: float) -> None:
        pass


class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # He initialization: better scaling for deep networks
        self.w = 0.1 * np.random.randn(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Perform the linear transformation: output = inp * W + b"""
        self.inp = inp
        self.out = np.dot(inp, self.w) + self.b
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backpropagate the gradients through this layer."""
        # Compute gradients for weights and biases
        self.dw = np.dot(self.inp.T, up_grad)  # Gradient wrt weights
        self.db = np.sum(up_grad, axis=0, keepdims=True)  # Gradient wrt biases
        # Compute gradient to propagate back (downstream)
        down_grad = np.dot(up_grad, self.w.T)
        return down_grad

    def step(self, lr: float) -> None:
        """Update the weights and biases using the gradients."""
        self.w -= lr * self.dw
        self.b -= lr * self.db


class ReLU(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """ReLU Activation: f(x) = max(0, x)"""
        self.inp = inp
        self.out = np.maximum(0, inp)
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for ReLU: derivative is 1 where input > 0, else 0."""
        down_grad = up_grad * (self.inp > 0)  # Efficient boolean indexing
        return down_grad


class Softmax(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Softmax Activation: f(x) = exp(x) / sum(exp(x))"""
        # Subtract max for numerical stability
        exp_values = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        self.out = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        """Backward pass for Softmax using the Jacobian matrix."""
        down_grad = np.empty_like(up_grad)
        for i in range(up_grad.shape[0]):
            single_output = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            down_grad[i] = np.dot(jacobian, up_grad[i])
        return down_grad


class Loss:
    def __init__(self):
        self.prediction = None
        self.target = None
        self.loss = None

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.forward(prediction, target)

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class CrossEntropy(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Cross-Entropy Loss for classification."""
        self.prediction = prediction
        self.target = target
        # Clip predictions to avoid log(0)
        clipped_pred = np.clip(prediction, 1e-12, 1.0)
        # Compute and return the loss
        self.loss = -np.mean(np.sum(target * np.log(clipped_pred), axis=1))
        return self.loss

    def backward(self) -> np.ndarray:
        """Gradient of Cross-Entropy Loss."""
        # Gradient wrt prediction (assuming softmax and one-hot targets)
        grad = -self.target / self.prediction / self.target.shape[0]
        return grad


class CNN:
    def __init__(self, layers: list[Layer], loss_fn: Loss, lr: float) -> None:
        """
        Convolutional Neural Network (CNN) class.
        Arguments:
        - layers: List of layers (e.g., Linear, ReLU, etc.).
        - loss_fn: Loss function object (e.g., CrossEntropy, MSE).
        - lr: Learning rate.
        """
        self.layers = layers
        self.loss_fn = loss_fn
        self.lr = lr

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        """Makes the model callable, equivalent to forward pass."""
        return self.forward(inp)

    def forward(self, inp: np.ndarray) -> np.ndarray:
        """Pass input through each layer sequentially."""
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Calculate the loss."""
        return self.loss_fn(prediction, target)

    def backward(self) -> None:
        """Perform backpropagation by propagating the gradient backwards through the layers."""
        up_grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            up_grad = layer.backward(up_grad)

    def update(self) -> None:
        """Update the parameters of each layer using the gradients and the learning rate."""
        for layer in self.layers:
            layer.step(self.lr)

    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int, kept_classes: list) -> np.ndarray:
        """Train the MLP over the given dataset for a number of epochs."""
        losses, accuracies = np.empty(epochs), np.empty(epochs)
        for epoch in (pbar := trange(epochs)):
            running_loss = 0.0
            correct = 0
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Chuyển đổi y_batch sang one-hot
                y_batch_onehot = onehot_encoder(y_batch, num_labels=len(kept_classes))

                # Forward pass
                prediction = self.forward(x_batch)

                # Compute loss với y_batch_onehot
                running_loss += self.loss(prediction, y_batch_onehot) * batch_size

                # Update correct
                correct += np.sum(np.argmax(prediction, axis=1) == y_batch)

                # Backward pass
                self.backward()

                # Update parameters
                self.update()

            # Normalize running loss và tính accuracy
            running_loss /= len(x_train)
            accuracy = 100 * correct / len(x_train)
            pbar.set_description(f"Loss: {running_loss:.3f} | Accuracy: {accuracy:.2f}% ")
            losses[epoch] = running_loss
            accuracies[epoch] = accuracy
        return losses, accuracies
    
def rotate_image(image, angle):
    """Xoay ảnh với góc xác định."""
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def load_images_from_folder(folder, label, image_size=(224, 224), augment=False):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = img / 255.0  # Chuẩn hóa về [0, 1]
            img = np.transpose(img, (2, 0, 1))  # Chuyển sang (channels, height, width)
            images.append(img)
            labels.append(label)
            
            # Áp dụng xoay ảnh nếu augment=True
            if augment:
                angles = [15, 30, -15, -30]  # Các góc xoay
                for angle in angles:
                    rotated_img = rotate_image(img.transpose(1, 2, 0), angle)  # Chuyển về (height, width, channels) để xử lý
                    rotated_img = np.transpose(rotated_img, (2, 0, 1))  # Chuyển lại về (channels, height, width)
                    images.append(rotated_img)
                    labels.append(label)
    
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)

def main():
    # Load dữ liệu với Data Augmentation
    meningioma_images, meningioma_labels = load_images_from_folder(
        'C:/Personal/final_graduate/data2/meningioma', 
        label=1, 
        augment=True  # Bật xoay ảnh
    )
    non_meningioma_images, non_meningioma_labels = load_images_from_folder(
        'C:/Personal/final_graduate/data2/normal', 
        label=0, 
        augment=True  # Bật xoay ảnh
    )
    
    # Kết hợp dữ liệu và shuffle
    X = np.concatenate([meningioma_images, non_meningioma_images], axis=0)
    y = np.concatenate([meningioma_labels, non_meningioma_labels], axis=0)
    
    # Shuffle dữ liệu
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # Phân chia train-test
    split_idx = int(0.8 * len(X))
    x_train, x_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Định nghĩa lớp và mô hình
    kept_classes = [0, 1]
    layers = [
        Conv2D(3, 4, 7, stride=3),
        ReLU(),
        MaxPool2D(),
        Conv2D(4, 4, 3, padding=1),
        ReLU(),
        MaxPool2D(),
        Flatten(),
        ReLU(),
        Linear(1296, len(kept_classes)),
        Softmax()
    ]
    
    # Huấn luyện mô hình
    model = CNN(layers, CrossEntropy(), lr=0.01)
    results = model.train(x_train, y_train, epochs=20, batch_size=16, kept_classes=kept_classes)
    
    # Vẽ kết quả
    plot_results(*results)
    
if __name__ == "__main__":
    main()

