import numpy as np
import torch

from ml_pipeline.hyperparams import Hyperparameters as hp


class CWSimulator():
    def __init__(self, dt: float, max_t: float, n: float, N_TRAJ: int, SEQUENCE_LENGTH: int):
        self.dt = dt
        self.max_t = max_t
        self.n = n
        self.N_TRAJ = N_TRAJ
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH

        self.times = np.arange(0, self.max_t+dt, dt) # Time vector

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

        if hp.N_INPUT_FEATURES == 3:
            return [x,y,z] + np.random.normal(0,2,3)
        elif hp.N_INPUT_FEATURES == 2:
            return [x,y] + np.random.normal(0,2,2)
        elif hp.N_INPUT_FEATURES == 1:
            return [x] + np.random.normal(0,2) 

    def simulate_trajectories(self) -> torch.Tensor:
        print(f'[INFO] Simulating {self.N_TRAJ} trajectories...')

        pos_is = -25 + 50 * np.random.random((self.N_TRAJ,3))
        vel_is = -2.5 + 5 * np.random.random((self.N_TRAJ,3))  
        
        # Concatenate pos_i and vel_i along the 2nd dimension
        X_is = np.concatenate((pos_is, vel_is), axis=1)

        X_t = np.asarray([[self.relative_position(X_i=X_i, t=t) for t in self.times] for X_i in X_is])

        X_t = torch.tensor(X_t).float()

        # if 2D, add a third dimension
        if len(X_t.shape) == 2:
            X_t = X_t.unsqueeze(2)
        print(f'trajectories shape: {X_t.shape}')

        return X_t
