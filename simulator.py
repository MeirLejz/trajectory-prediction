import numpy as np
import matplotlib.pyplot as plt
import torch

# simulation parameters
dt = 1 # seconds. Time step
T = 60*60 # seconds. Period of the orbit
n = 2*np.pi/T # rad/s. Orbital rate
N_TRAJ = 1 # Number of trajectories
SEQUENCE_LENGTH = 10 # Number of time steps to use for prediction
max_t = T # seconds. Maximum time to simulate

class CWSimulator():
    def __init__(self, dt: float=dt, max_t: float=max_t, n: float=n, N_TRAJ: int=N_TRAJ, SEQUENCE_LENGTH: int=SEQUENCE_LENGTH):
        self.dt = dt
        self.max_t = max_t
        self.n = n
        self.N_TRAJ = N_TRAJ
        self.times = torch.arange(0, self.max_t+dt, dt) # Time vector
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH

    def relative_position(self, X_i: np.ndarray, t: float) -> list[float]:
        x_0, y_0, z_0 = X_i[0:3]
        vx_0, vy_0, vz_0 = X_i[3:6]

        x = (
            (2 * vy_0 / self.n - 3 * x_0) * np.cos(self.n * t)
            + vx_0 / self.n * np.sin(self.n * t)
            + (4 * x_0 - 2 * vy_0 / self.n)
        )
        y = (
            (4 * vy_0 / self.n - 6 * x_0) * np.sin(self.n * t)
            - 2 * vx_0 / self.n * np.cos(self.n * t)
            + (6 * self.n * x_0 - 3 * vy_0) * t
            + (y_0 + 2 * vx_0 / self.n)
        )
        z = z_0 * np.cos(self.n * t) + vz_0 / self.n * np.sin(self.n * t)

        return [x,y,z]

    def simulate_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        pos_is =  5 * np.ones((self.N_TRAJ,3)) # -0.5 + 1 * np.random.random((self.N_TRAJ,3))
        vel_is = -0.05 + 0.1 * np.zeros((self.N_TRAJ,3))  #-.05 + 0.1*np.random.random((self.N_TRAJ,3)) # 0.1 * np.ones((self.N_TRAJ,3))  #

        # Concatenate pos_i and vel_i along the 2nd dimension
        X_is = np.concatenate((pos_is, vel_is), axis=1)


        X_t = np.asarray([[self.relative_position(X_i=X_i, t=t) for t in self.times] for X_i in X_is])
        # , subplot_kw={'projection': '3d'}
        # fig, ax = plt.subplots(1,1)
        # for j in range(N_TRAJ):

        #     x, y, z = X_t[j,:,:].T

            # , z, , z[0], ,0
        #     ax.plot(x, y, linewidth=1)
        #     ax.scatter(x[0],y[0], color='r')
        # ax.scatter(0,0, color='k')

        # plt.show()
        print(f'X_t shape: {X_t.shape}')
        return X_t

    def create_training_data(self, X_t: np.ndarray=0) -> tuple[np.ndarray, np.ndarray]:

        # rnn_input = np.zeros((self.N_TRAJ * (len(self.times) - self.SEQUENCE_LENGTH - 1), self.SEQUENCE_LENGTH, 3))
        # rnn_output = np.zeros((self.N_TRAJ * (len(self.times) - self.SEQUENCE_LENGTH - 1), 3))

        # for j in range(self.N_TRAJ):
        #     for k in range(len(self.times) - self.SEQUENCE_LENGTH - 1):
        #         rnn_input[j * (len(self.times) - self.SEQUENCE_LENGTH - 1) + k, :, :] = X_t[j, k:k + self.SEQUENCE_LENGTH, :]
        #         rnn_output[j * (len(self.times) - self.SEQUENCE_LENGTH - 1) + k, :] = X_t[j, k + self.SEQUENCE_LENGTH, :]

        x = (
            (2 * 0.1 / self.n - 3 * 1) * torch.cos(self.n * self.times)
            + 0.1 / self.n * torch.sin(self.n * self.times)
            + (4 * 1 - 2 * 0.1 / self.n)
        )
        print(f'x shape: {x.shape}')

        rnn_input = torch.zeros(((len(self.times) - self.SEQUENCE_LENGTH - 1), self.SEQUENCE_LENGTH))
        rnn_output = torch.zeros((len(self.times) - self.SEQUENCE_LENGTH - 1))

        for k in range(len(self.times) - self.SEQUENCE_LENGTH - 1):
            rnn_input[k, :] = x[k:k + self.SEQUENCE_LENGTH]
            rnn_output[k] = x[k + self.SEQUENCE_LENGTH]

        # sig = 5 * torch.cos(2*torch.arange(0,11)) + 4 * torch.sin(3.2*torch.arange(0,11))
        # sig2 = 5 * torch.cos(2*torch.arange(1,12)) + 4 * torch.sin(3.2*torch.arange(1,12))
        # sig_out = 5 * torch.cos(torch.tensor([2*11])) + 4 * torch.sin(torch.tensor([3.2*11]))
        # sig_out2 = 5 * torch.cos(torch.tensor([2*12])) + 4 * torch.sin(torch.tensor([3.2*12]))
        # rnn_input = torch.cat([sig[None,:], sig2[None,:]], dim=0).unsqueeze(-1).float()
        # rnn_output = torch.cat([sig_out[None,:], sig_out2[None,:]], dim=0).unsqueeze(-1).float()


        print(f'rnn_input shape: {rnn_input.shape}, rnn_output shape: {rnn_output.shape}')
        return rnn_input.unsqueeze(-1).float(), rnn_output.unsqueeze(-1).float()