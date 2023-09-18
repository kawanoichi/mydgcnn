"""pytorchがgpuを認識できているか確認する.

@author kawanoichi
実行コマンド
$ python3 enviroment_check/check_torch_gpu.py
"""
import torch

print("torch version ", torch.__version__)

print("GPU確認", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(torch.cuda.current_device())
