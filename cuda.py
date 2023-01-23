import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
torch.cuda.set_device(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
