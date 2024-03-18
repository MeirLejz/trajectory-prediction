import numpy as np
import torch
from torch.utils.data import Dataset

class CWTrajDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor, transform=None, target_transform=None):
        self.inputs = inputs
        self.outputs = outputs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = torch.Tensor(self.inputs[idx,:])
        if self.transform:
            input = self.transform(input)

        output = torch.Tensor(self.outputs[idx])
        if self.target_transform:
            output = self.target_transform(output)
        return input, output

class _CWTrajDataset(Dataset):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray, transform=None, target_transform=None):
        self.inputs = inputs
        self.outputs = outputs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx: int):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        input = torch.Tensor(self.inputs[idx,:,:])
        if self.transform:
            input = self.transform(input)

        output = torch.Tensor(self.outputs[idx,:])
        if self.target_transform:
            output = self.target_transform(output)
        return input, output