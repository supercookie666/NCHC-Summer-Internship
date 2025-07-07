import torch
import torch.nn as nn
import torch.optim as optim

# 建立一個超簡單的神經網路
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 輸入層：10維 → 隱藏層：50維
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)   # 隱藏層：50維 → 輸出層：1維（回歸）

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型、損失函數和優化器
model = SimpleNet()
criterion = nn.MSELoss()  # 均方誤差損失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 隨機梯度下降

# 隨機生成假資料（100筆資料，每筆10維特徵）
inputs = torch.randn(100, 10)  # 輸入資料
targets = torch.randn(100, 1)  # 目標值（回歸）

# 訓練模型
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()          # 清除上一步的梯度
    outputs = model(inputs)        # 前向傳播
    loss = criterion(outputs, targets)  # 計算損失
    loss.backward()                # 反向傳播
    optimizer.step()               # 更新權重

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("✅ 訓練完成！")
