import torch

# 建一個隨機形狀為 3x4 的張量
tensor = torch.rand(3, 4)

#— 雙語註解 🌱
print(f"Shape of tensor: {tensor.shape}")         # Tensor 的尺寸
print(f"Datatype of tensor: {tensor.dtype}")      # Tensor 的資料型態
print(f"Device tensor is stored on: {tensor.device}")  # Tensor 存放在哪（CPU/GPU）