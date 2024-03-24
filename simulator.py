import numpy as np
import matplotlib.pyplot as plt
import torch

# simulation parameters
dt = 1 # seconds. Time step
T = 60*60 # seconds. Period of the orbit
n = 2*np.pi/T # rad/s. Orbital rate
N_TRAJ = 10 # Number of trajectories
SEQUENCE_LENGTH = 40 # Number of time steps to use for prediction
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

        return x # [x,y,z]

    def simulate_trajectories(self) -> np.ndarray:
        pos_is = 5 + 1 * np.random.random((self.N_TRAJ,3))
        vel_is = 0.1 * np.zeros((self.N_TRAJ,3))  
        
        # Concatenate pos_i and vel_i along the 2nd dimension
        X_is = np.concatenate((pos_is, vel_is), axis=1)

        X_t = np.asarray([[self.relative_position(X_i=X_i, t=t) for t in self.times] for X_i in X_is])

        return X_t
