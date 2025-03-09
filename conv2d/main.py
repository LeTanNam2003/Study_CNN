# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, random_split
# import numpy as np
# import cv2
# from module import Conv2D  # Import your custom Conv2D class
# # Đo thời gian tiền xử lý dữ liệu
# start_time = time.perf_counter()

# # Data Preprocessing and Augmentation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])  
# ])

# dataset = datasets.ImageFolder('C:/Personal/final_graduate/data', transform=transform)

# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# data_preprocessing_time = time.perf_counter() - start_time
# print(f"⏳ Thời gian tiền xử lý dữ liệu: {data_preprocessing_time:.4f} giây")


# # class VGG16(nn.Module):
# #     def __init__(self, num_classes=1):  
# #         super(VGG16, self).__init__()

# #         self.features = nn.Sequential(
# #             nn.Conv2d(3, 64, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(64, 64, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),  

# #             nn.Conv2d(64, 128, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(128, 128, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),  

# #             nn.Conv2d(128, 256, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),  

# #             nn.Conv2d(256, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),  

# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),  

# #             nn.AdaptiveAvgPool2d((7, 7))  
# #         )

# #         self.classifier = nn.Sequential(
# #             nn.Linear(512 * 7 * 7, 4096),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(0.5),
# #             nn.Linear(4096, 4096),
# #             nn.ReLU(inplace=True),
# #             nn.Dropout(0.5),
# #             nn.Linear(4096, num_classes)  
# #         )

# #     def forward(self, x):
# #         x = self.features(x)
# #         x = torch.flatten(x, start_dim=1)  
# #         x = self.classifier(x)
# #         return x
# class VGG16(nn.Module):
#     def __init__(self, num_classes=1):
#         super(VGG16, self).__init__()

#         self.conv1 = Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = Conv2D(64, 64, kernel_size=3, stride=1, padding=1)

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Dùng pooling của PyTorch

#         self.conv3 = Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv4 = Conv2D(128, 128, kernel_size=3, stride=1, padding=1)

#         self.conv5 = Conv2D(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv6 = Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv7 = Conv2D(256, 256, kernel_size=3, stride=1, padding=1)

#         self.conv8 = Conv2D(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv9 = Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv10 = Conv2D(512, 512, kernel_size=3, stride=1, padding=1)

#         self.conv11 = Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv12 = Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv13 = Conv2D(512, 512, kernel_size=3, stride=1, padding=1)

#         self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(4096, num_classes)
#         )

#     def forward(self, x):
#         x = self.conv1.forward(x)
#         x = self.conv2.forward(x)
#         x = self.pool(x)

#         x = self.conv3.forward(x)
#         x = self.conv4.forward(x)
#         x = self.pool(x)

#         x = self.conv5.forward(x)
#         x = self.conv6.forward(x)
#         x = self.conv7.forward(x)
#         x = self.pool(x)

#         x = self.conv8.forward(x)
#         x = self.conv9.forward(x)
#         x = self.conv10.forward(x)
#         x = self.pool(x)

#         x = self.conv11.forward(x)
#         x = self.conv12.forward(x)
#         x = self.conv13.forward(x)
#         x = self.pool(x)

#         x = self.adaptive_pool(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.classifier(x)

#         return x



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = VGG16(num_classes=1).to(device)

# criterion = nn.BCEWithLogitsLoss()  
# optimizer = optim.Adam(model.parameters(), lr=0.0001)  
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# num_epochs = 1
# patience = 5
# best_val_loss = float('inf')
# epochs_without_improvement = 0

# # Training Loop
# for epoch in range(num_epochs):
#     start_epoch_time = time.perf_counter()  # Đo thời gian của epoch

#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     start_train_time = time.perf_counter()  # Đo thời gian training
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         predictions = (torch.sigmoid(outputs) > 0.5).float()  
#         correct += (predictions == labels).sum().item()
#         total += labels.size(0)
#         running_loss += loss.item()

#     train_time = time.perf_counter() - start_train_time
#     train_loss = running_loss / len(train_loader)
#     train_accuracy = 100 * correct / total

#     start_val_time = time.perf_counter()  # Đo thời gian validation
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()

#             predictions = (torch.sigmoid(outputs) > 0.5).float()
#             correct += (predictions == labels).sum().item()
#             total += labels.size(0)

#     val_time = time.perf_counter() - start_val_time
#     val_loss /= len(val_loader)
#     val_accuracy = 100 * correct / total

#     epoch_time = time.perf_counter() - start_epoch_time  # Tổng thời gian của epoch

#     print(f"⏳ Epoch [{epoch+1}/{num_epochs}] | Train Time: {train_time:.4f}s | Validation Time: {val_time:.4f}s | Total: {epoch_time:.4f}s")
#     print(f"📉 Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%\n")

#     scheduler.step(val_loss)  

#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         epochs_without_improvement = 0
#         torch.save(model.state_dict(), 'best_model_vgg16_v4_4.pth')
#     else:
#         epochs_without_improvement += 1

#     if epochs_without_improvement >= patience:
#         print(f"⚠️ Early stopping triggered. No improvement for {patience} epochs.")
#         break

# # Đo thời gian inference
# def predict_and_segment(image):
#     start_predict_time = time.perf_counter()
    
#     model_input = transform(image).unsqueeze(0).to(device)
#     output = model(model_input)
#     prediction = torch.sigmoid(output).item()
    
#     predict_time = time.perf_counter() - start_predict_time
#     print(f"⏳ Thời gian dự đoán: {predict_time:.4f} giây")

#     if prediction > 0.5:  
#         segmented_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
#         _, binary = cv2.threshold(segmented_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         return binary
#     else:
#         return None

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import json
import matplotlib.pyplot as plt
import numpy as np
from module import Conv2D  # Import lớp Conv2D bạn tự cài đặt

# ==============================
# 1️⃣ Tiền xử lý dữ liệu
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = datasets.ImageFolder('C:/Personal/final_graduate/data', transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ==============================
# 2️⃣ Định nghĩa mô hình VGG16
# ==============================
class VGG16(nn.Module):
    def __init__(self, num_classes=1):
        super(VGG16, self).__init__()

        self.conv1 = Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2D(64, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = Conv2D(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv5 = Conv2D(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = Conv2D(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = Conv2D(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv8 = Conv2D(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = Conv2D(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = Conv2D(512, 512, kernel_size=3, stride=1, padding=1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool(x)

        x = self.conv3.forward(x)
        x = self.conv4.forward(x)
        x = self.pool(x)

        x = self.conv5.forward(x)
        x = self.conv6.forward(x)
        x = self.conv7.forward(x)
        x = self.pool(x)

        x = self.conv8.forward(x)
        x = self.conv9.forward(x)
        x = self.conv10.forward(x)
        x = self.pool(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

# ==============================
# 3️⃣ Khởi tạo mô hình, loss, optimizer
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(num_classes=1).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# ==============================
# 4️⃣ Training Loop + Lưu loss & accuracy
# ==============================
num_epochs = 1  
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    scheduler.step(val_loss)

# Lưu loss & accuracy vào file JSON
train_history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accuracies": train_accuracies,
    "val_accuracies": val_accuracies
}

with open("train_history.json", "w") as f:
    json.dump(train_history, f)

print("✅ Train history đã được lưu vào train_history.json!")

# ==============================
# 5️⃣ Load lại loss & accuracy
# ==============================
with open("train_history.json", "r") as f:
    train_history = json.load(f)

print("📊 Train Loss:", train_history["train_losses"])
print("📈 Train Accuracy:", train_history["train_accuracies"])
print("📉 Validation Loss:", train_history["val_losses"])
print("✅ Validation Accuracy:", train_history["val_accuracies"])

# ==============================
# 6️⃣ Vẽ biểu đồ loss & accuracy
# ==============================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_history["train_losses"], label="Train Loss")
plt.plot(train_history["val_losses"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Graph")

plt.subplot(1, 2, 2)
plt.plot(train_history["train_accuracies"], label="Train Acc")
plt.plot(train_history["val_accuracies"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Graph")

plt.show()
