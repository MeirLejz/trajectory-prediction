from simulator import CWSimulator
import matplotlib.pyplot as plt
import torch
import numpy as np
import pdb
from hyperparams import Hyperparameters as hp


def load_model(path: str = './output/model.pth') -> torch.nn.Module:
    model = torch.load(path)
    return model

def main():

    # Load the model
    model = load_model(path='./output/model.pth')
    model.eval()

    simulator = CWSimulator(N_TRAJ=1)
    # Create test data from simulator.py
    trajectories = simulator.simulate_trajectories()

    trajectories = torch.tensor(trajectories).float()

    # if 2D, add a third dimension
    if len(trajectories.shape) == 2:
        trajectories = trajectories.unsqueeze(2)
    print(f'trajectories shape: {trajectories.shape}')

    seq_length = simulator.SEQUENCE_LENGTH


    ynn = np.zeros((simulator.N_TRAJ, len(simulator.times), hp.N_INPUT_FEATURES))
    ynn[:, 0:seq_length, :] = trajectories[:, 0:seq_length, :]
    print(f'lenght of times: {len(simulator.times)}')
    
    with torch.no_grad():
        for j in range(simulator.N_TRAJ):
            for k in range(seq_length, len(simulator.times) - hp.N_FUTURE_STEPS, hp.N_FUTURE_STEPS):
                # rnn_input = torch.tensor(ynn[j, k - seq_length:k, :]).float().unsqueeze(0)

                rnn_input = trajectories[j, k - seq_length:k, :].unsqueeze(0)

                output = model(rnn_input).numpy()
                # print(f'output shape: {np.shape(output)}')
                ynn[j, k:k+hp.N_FUTURE_STEPS, :] = output

    ax = plt.figure().add_subplot(projection=None if hp.N_INPUT_FEATURES == 1 else '3d')
    for j in range(simulator.N_TRAJ):

        if hp.N_INPUT_FEATURES == 1:
            
            x = trajectories[j,:,:].squeeze().numpy()
            ax.plot(simulator.times, x, linewidth=1,marker='o')
            ax.plot(x[0], color='r')
            x_ = ynn[j,:,:].squeeze()
            ax.plot(simulator.times, x_, linewidth=1, linestyle='--', marker='x')
            ax.plot(x_[seq_length], color='m')
            ax.plot(0,0, color='k')

        elif hp.N_INPUT_FEATURES == 2:
            x, y = trajectories[j,:,:].squeeze().T.numpy()
            ax.plot(x, y, linewidth=1,marker='o')
            ax.scatter(x[0], y[0], color='r')
            x_, y_ = ynn[j,:,:].squeeze().T
            ax.plot(x_, y_, linewidth=1, linestyle='--', marker='x')
            ax.scatter(x_[seq_length], y_[seq_length], color='m')
            ax.scatter(0,0, color='k')

        elif hp.N_INPUT_FEATURES == 3:
            x, y, z = trajectories[j,:,:].squeeze().T.numpy()
            ax.plot(x, y, z, linewidth=1,marker='o')
            ax.scatter(x[0], y[0], z[0], color='r')
            x_, y_, z_ = ynn[j,:,:].squeeze().T
            ax.plot(x_, y_, z_, linewidth=1, linestyle='--', marker='x')
            ax.scatter(x_[seq_length], y_[seq_length], z_[seq_length], color='m')
            ax.scatter(0,0,0, color='k')

    plt.show()
    pdb.set_trace()
                
if __name__ == '__main__':
    main()