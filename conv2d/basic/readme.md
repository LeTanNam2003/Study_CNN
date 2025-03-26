# **Phân Tích Thuật Toán CNN Thủ Công**

---

## **1️⃣ Load Data & Preprocessing**
📌 **Chức năng:**  
- Đọc ảnh từ thư mục, chuyển thành ảnh grayscale và resize.  
- Chuẩn hóa pixel về `[0, 1]` để tăng hiệu suất huấn luyện.  
- Chia tập train/test theo tỷ lệ `80/20`.  

🔍 **Phân tích thuật toán:**  
- Duyệt qua thư mục chứa ảnh để lấy danh sách nhãn (`class_labels`).  
- Đọc ảnh, resize về kích thước `(img_size, img_size)`.  
- Biến đổi ảnh thành mảng NumPy, cùng với nhãn số hóa (`label_idx`).  
- **Độ phức tạp:** \(O(N)\) (với \(N\) là số lượng ảnh).  

---

## **2️⃣ Forward Propagation (Lan truyền xuôi)**
📌 **Chức năng:**  
- Tính toán từng lớp của mạng nơ-ron theo thứ tự:  
  **Conv2D → ReLU → Conv2D → ReLU → Flatten → Dense → Softmax**  

🔍 **Phân tích thuật toán:**  
### **➤ Convolution Layer**
```python
 def conv2d(X, kernel):
    h, w = X.shape
    kh, kw = kernel.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    output = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(X[i:i+kh, j:j+kw] * kernel)
    
    return output
```
- **Công thức:**  
  \[
  y[i, j] = \sum_{m=0}^{kh-1} \sum_{n=0}^{kw-1} X[i+m, j+n] \cdot K[m, n]
  \]
- **Độ phức tạp:** \(O(H \cdot W \cdot kH \cdot kW)\)  
  (Với \(H, W\) là kích thước ảnh, \(kH, kW\) là kích thước kernel)  
- **Cải tiến:**  
  - Dùng **im2col + phép nhân ma trận** để tăng tốc.  

### **➤ ReLU Activation**
```python
 def relu(X):
    return np.maximum(0, X)
```
- **Công thức:**  
  \[
  f(x) = \max(0, x)
  \]
- **Độ phức tạp:** \(O(N)\), chạy trên từng phần tử.  

### **➤ Fully Connected (Dense)**
```python
 def dense(X, weights, bias):
    return np.dot(X, weights) + bias
```
- **Công thức:**  
  \[
  y = XW + b
  \]
- **Độ phức tạp:** \(O(NM)\).  

### **➤ Softmax + Loss Function**
```python
 def softmax(X):
    exp_X = np.exp(X - np.max(X))
    return exp_X / np.sum(exp_X)

def cross_entropy_loss(y_true, y_pred):
    return -np.log(y_pred[y_true])
```
- **Softmax** tính xác suất dự đoán.  
- **Cross-Entropy Loss** đo độ sai khác giữa dự đoán và thực tế.  
- **Độ phức tạp:** \(O(N)\).  

---

## **3️⃣ Backpropagation (Lan truyền ngược)**
📌 **Chức năng:**  
- Tính toán gradient của từng lớp và cập nhật trọng số.  

🔍 **Phân tích thuật toán:**  
### **➤ Backpropagation trong Convolution**
```python
 def conv2d_backward(input, grad_output, kernel, lr):
    grad_kernel = np.zeros_like(kernel)
    for i in range(grad_output.shape[0]):
        for j in range(grad_output.shape[1]):
            grad_kernel += grad_output[i, j] * input[i:i+kernel.shape[0], j:j+kernel.shape[1]]
    kernel -= lr * grad_kernel
```
- **Gradient của Kernel:**  
  \[
  \frac{\partial L}{\partial K} = \sum \text{(Gradient Output) } \cdot \text{(Region Input)}
  \]
- **Độ phức tạp:** \(O(H \cdot W \cdot kH \cdot kW)\).  

---

## **4️⃣ Training Loop**
📌 **Chức năng:**  
- Lặp `100` epochs.  
- Cập nhật trọng số sau mỗi batch size (`32` ảnh).  
- Tính Accuracy & Loss sau mỗi epoch.  

🔍 **Phân tích thuật toán:**  
```python
 for epoch in range(100):
    total_loss = 0
    correct = 0
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_Y = Y_train[i:i+batch_size]
        
        # Forward Pass
        ...
        
        # Backpropagation
        ...
        
    accuracy = correct / len(X_train)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(X_train)}, Accuracy: {accuracy}")
```
- **Độ phức tạp:** \(O(E \cdot \frac{N}{B} \cdot C)\)  
  (E: Epochs, N: Số mẫu, B: Batch Size, C: Số phép tính của model).  

---

## **5️⃣ Đánh giá hiệu suất**
📌 **Quan sát từ kết quả train:**  
- **Accuracy dần tăng**, nghĩa là mô hình đang học tốt.  
- **Loss không giảm đều**, có thể do:  
  - Learning rate cần giảm dần (schedule).  
  - Dữ liệu chưa đủ lớn.  
  - Model còn đơn giản, cần nhiều layers hơn.  

📌 **Gợi ý cải thiện:**  
- **Thêm Batch Normalization** sau mỗi Conv2D.  
- **Sử dụng Adam Optimizer** thay vì SGD.  
- **Tăng số filters trong Convolution** để học đặc trưng tốt hơn.  

---

## **🔥 Kết luận**
- **Mạng CNN thủ công này hoạt động tốt nhưng còn đơn giản.**  
- **Có thể tối ưu tốc độ bằng im2col + GEMM** thay vì vòng lặp trong `conv2d`.  
- **Mô hình có thể đạt accuracy tốt hơn với dropout + batchnorm.**  


