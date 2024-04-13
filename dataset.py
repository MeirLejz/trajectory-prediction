import numpy as np
import torch
from torch.utils.data import Dataset


class CWTrajDataset(Dataset):
    def __init__(self, trajectories: torch.Tensor, sequence_len: int, transform=None, target_transform=None, n_input_features: int=1, future_len: int=5):
        
        print(f'[INFO] Creating dataset...')
        # self.trajectories = trajectories
        self.n_traj = trajectories.shape[0]
        self.traj_len = trajectories.shape[1]
        self.sequence_len = sequence_len
        self.n_input_features = n_input_features
        self.future_len = future_len    

        self.inputs = torch.zeros((self.n_traj * (self.traj_len - self.sequence_len - 1), self.sequence_len, n_input_features))
        self.outputs = torch.zeros((self.n_traj * (self.traj_len - self.sequence_len - 1), self.future_len, n_input_features))

        maxes, _ = torch.max(abs(trajectories), axis=1)
        self.bounds, _ = torch.max(maxes, axis=0)

        self.trajectories = trajectories / self.bounds

        for j in range(self.n_traj):
            for k in range(self.traj_len - self.sequence_len - self.future_len - 1):
                self.inputs[j * (self.traj_len - self.sequence_len - self.future_len - 1) + k, :, :] = self.trajectories[j, k:(k + self.sequence_len), :]
                self.outputs[j * (self.traj_len - self.sequence_len - self.future_len - 1) + k, :, :] = self.trajectories[j, (k + self.sequence_len): (k + self.sequence_len + self.future_len), :]
                
        self.transform = transform
        self.target_transform = target_transform

        print(f'[INFO] Training dataset size: {len(self)}')

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input = self.inputs[idx,:,:]
        if self.transform:
            input = self.transform(input)

        output = self.outputs[idx,:,:]
        if self.target_transform:
            output = self.target_transform(output)
        return input, output
