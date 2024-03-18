from simulator import CWSimulator
import matplotlib.pyplot as plt
import torch
import numpy as np


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
    seq_length = simulator.SEQUENCE_LENGTH
    ynn = np.zeros((simulator.N_TRAJ, len(simulator.times), 3))
    ynn[:, 0:seq_length, :] = trajectories[:, 0:seq_length, :]
    print(f'lenght of times: {len(simulator.times)}')
    with torch.no_grad():
        for j in range(simulator.N_TRAJ):
            for k in range(seq_length, len(simulator.times)):
                rnn_input = torch.tensor(ynn[j, k - seq_length:k, :]).float().unsqueeze(0)
                # import pdb; pdb.set_trace()

                # print(f'rnn_input shape: {rnn_input.shape}')
                output = model(rnn_input).squeeze().numpy()
                # print(f'output shape: {np.shape(output)}')
                ynn[j, k, :] = output

    
    fig, ax = plt.subplots(1,1,subplot_kw={'projection': '3d'})
    for j in range(simulator.N_TRAJ):

        x, y, z = trajectories[j,:,:].T
        ax.plot(x, y, z, linewidth=1)
        ax.scatter(x[0], y[0], z[0], color='r')
        x_, y_, z_ = ynn[j,:,:].T
        ax.plot(x_, y_, z_, linewidth=1, linestyle='--')
        ax.scatter(x_[seq_length], y_[seq_length], z_[seq_length], color='m')

    ax.scatter(0,0, color='k')

    plt.show()
    import pdb; pdb.set_trace()
                
if __name__ == '__main__':
    main()