# import torch
  
# print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
  
# # Storing ID of current CUDA device
# cuda_id = torch.cuda.current_device()
# print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        
# print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
import torch
from torch import nn
m = nn.Softmax(dim=1)
output = nn.Softmax(dim=-1)(torch.tensor([1,2,3,4], dtype=torch.float))
# input = torch.randn(1, 3)
# output = m(input)
print(output)