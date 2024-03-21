import numpy as np
import torch
from torch.utils.data import Dataset


class CWTrajDataset(Dataset):
    def __init__(self, trajectories: torch.Tensor, sequence_len: int, transform=None, target_transform=None, n_input_features: int=1):
        
        self.trajectories = trajectories
        self.n_traj = trajectories.shape[0]
        self.traj_len = trajectories.shape[1]
        self.sequence_len = sequence_len
        self.n_input_features = n_input_features

        self.inputs = torch.zeros((self.n_traj * (self.traj_len - self.sequence_len - 1), self.sequence_len, n_input_features))
        self.outputs = torch.zeros((self.n_traj * (self.traj_len - self.sequence_len - 1), n_input_features))

        for j in range(self.n_traj):
            for k in range(self.traj_len - self.sequence_len - 1):
                self.inputs[j * (self.traj_len - self.sequence_len - 1) + k, :, :] = self.trajectories[j, k:k + self.sequence_len, :]
                self.outputs[j * (self.traj_len - self.sequence_len - 1) + k, :] = self.trajectories[j, k + self.sequence_len, :]
                
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.inputs[idx,:,:]
        if self.transform:
            input = self.transform(input)

        output = self.outputs[idx,:]
        if self.target_transform:
            output = self.target_transform(output)
        return input, output


class _CWTrajDataset(Dataset):
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