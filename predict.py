from simulator import CWSimulator
import matplotlib.pyplot as plt
import torch
import numpy as np
import pdb


def load_model(path: str = './output/model.pth') -> torch.nn.Module:
    model = torch.load(path)
    return model

def main():

    # Load the model
    model = load_model(path='./output/model.pth')
    model.eval()

    # simulator = CWSimulator(N_TRAJ=1)
    # Create test data from simulator.py
    # trajectories = simulator.simulate_trajectories()

    dt = 10 # seconds. Time step
    T = 60*60 # seconds. Period of the orbit
    n = 2*np.pi/T # rad/s. Orbital rate
    N_TRAJ = 1 # Number of trajectories
    SEQUENCE_LENGTH = 10 # Number of time steps to use for prediction
    max_t = T # seconds. Maximum time to simulate
    times = torch.arange(0, max_t+dt, dt)
    x = (
        (2 * 0.1 / n - 3 * 1) * torch.cos(n * times)
        + 0.1 / n * torch.sin(n * times)
        + (4 * 1 - 2 * 0.1 / n)
    )   



    seq_length = SEQUENCE_LENGTH
    ynn = np.zeros((len(times)))
    ynn[0:seq_length] = x[0:seq_length]
    print(f'lenght of times: {len(times)}')
    with torch.no_grad():
        for k in range(seq_length, len(times)):
            rnn_input = torch.tensor(ynn[k - seq_length:k]).float().unsqueeze(1).unsqueeze(0)
            # import pdb; pdb.set_trace()
            # print(f'rnn_input shape: {rnn_input.shape}')
            # pdb.set_trace()

            output = model(rnn_input).squeeze().numpy()
            # print(f'output shape: {np.shape(output)}')
            ynn[k] = output

    
    fig, ax = plt.subplots(1,1,subplot_kw={'projection': '3d'})

    ax.plot(times, x, linewidth=1)
    ax.scatter(0, x[0], color='r')
    ax.plot(times, ynn, linewidth=1, linestyle='--')
    ax.scatter(seq_length, ynn[seq_length], color='m')


    plt.show()


    # ynn = np.zeros((simulator.N_TRAJ, len(simulator.times), 3))
    # ynn[:, 0:seq_length, :] = trajectories[:, 0:seq_length, :]
    # print(f'lenght of times: {len(simulator.times)}')
    # with torch.no_grad():
    #     for j in range(simulator.N_TRAJ):
    #         for k in range(seq_length, len(simulator.times)):
    #             rnn_input = torch.tensor(ynn[j, k - seq_length:k, :]).float().unsqueeze(0)
    #             # import pdb; pdb.set_trace()

    #             # print(f'rnn_input shape: {rnn_input.shape}')
    #             output = model(rnn_input).squeeze().numpy()
    #             # print(f'output shape: {np.shape(output)}')
    #             ynn[j, k, :] = output

    
    # fig, ax = plt.subplots(1,1,subplot_kw={'projection': '3d'})
    # for j in range(simulator.N_TRAJ):

    #     x, y, z = trajectories[j,:,:].T
    #     ax.plot(x, y, z, linewidth=1)
    #     ax.scatter(x[0], y[0], z[0], color='r')
    #     x_, y_, z_ = ynn[j,:,:].T
    #     ax.plot(x_, y_, z_, linewidth=1, linestyle='--')
    #     ax.scatter(x_[seq_length], y_[seq_length], z_[seq_length], color='m')

    # ax.scatter(0,0, color='k')

    # plt.show()
    pdb.set_trace()
                
if __name__ == '__main__':
    main()