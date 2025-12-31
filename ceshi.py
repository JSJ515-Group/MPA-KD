import torch

print("\n===== CUDA状态检测 =====")
print(f"PyTorch版本：{torch.__version__}")
print(f"CUDA是否可用：{'✅可用' if torch.cuda.is_available() else '❌不可用'}")
print(f"检测到的GPU数量：{torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU设备：{torch.cuda.current_device()}")
    print(f"设备名称：{torch.cuda.get_device_name(0)}")
else:
    print("（未检测到可用GPU设备）")