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
            for k in range(1): # range(seq_length, len(simulator.times) - hp.N_FUTURE_STEPS, hp.N_FUTURE_STEPS):
                # rnn_input = torch.tensor(ynn[j, k - seq_length:k, :]).float().unsqueeze(0)
                rnn_input = trajectories[j, k - seq_length:k, :].unsqueeze(0)

                output = model(rnn_input).numpy()
                # print(f'output shape: {np.shape(output)}')
                ynn[j, k:k+hp.N_FUTURE_STEPS, :] = output


    _, ax = plt.subplots(2,1)
    for j in range(simulator.N_TRAJ):

        x = trajectories[j,:,:].squeeze().numpy()
        ax[0].plot(simulator.times, x, linewidth=1,marker='o')
        # ax.scatter(x[0], color='r')
        x_ = ynn[j,:,:].squeeze()
        ax[0].plot(simulator.times, x_, linewidth=1, linestyle='--', marker='x')
        # ax.scatter(x_[seq_length], color='m')

    ax[0].scatter(0,0, color='k')

    ax[1].plot(simulator.times, x - x_, linewidth=1, marker='o', markersize=0.1)


    plt.show()
    pdb.set_trace()
                
if __name__ == '__main__':
    main()