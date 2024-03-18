import torch, numpy as np

# simulation parameters
dt = 0.1 # seconds. Time step
T = 60*60 # seconds. Period of the orbit
n = 2*np.pi/T # rad/s. Orbital rate
N_TRAJ = 1 # Number of trajectories
SEQUENCE_LENGTH = 10 # Number of time steps to use for prediction


x = (
    (2 * 0.1 / n - 3 * 1) * torch.cos(n * torch.tensor(times))
    + 0.1 / n * torch.sin(n * torch.tensor(times))
    + (4 * 1 - 2 * 0.1 / n)
)


# TRAIN_SIZE = int(0.7 * rnn_input.shape[0])
sig = 5 * torch.cos(2*torch.arange(0,11)) + 4 * torch.sin(3.2*torch.arange(0,11))
sig2 = 5 * torch.cos(2*torch.arange(1,12)) + 4 * torch.sin(3.2*torch.arange(1,12))
sig_out = 5 * torch.cos(torch.tensor([2*11])) + 4 * torch.sin(torch.tensor([3.2*11]))
sig_out2 = 5 * torch.cos(torch.tensor([2*12])) + 4 * torch.sin(torch.tensor([3.2*12]))
rnn_input = torch.cat([sig[None,:], sig2[None,:]], dim=0).unsqueeze(-1).float()
rnn_output = torch.cat([sig_out[None,:], sig_out2[None,:]], dim=0).unsqueeze(-1).float()