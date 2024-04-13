from dataclasses import dataclass
import torch

@dataclass
class Hyperparameters:

    # training flow hyperparameters
    INIT_LR: float = 1e-3
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    MILESTONES = [30, 60, 90]

    # datasets hyperparameters
    TRAIN_SPLIT: float = 0.75
    VAL_SPLIT: float = 1 - TRAIN_SPLIT
    
    # model hyperparameters
    HIDDEN_SIZE: int = 64
    NUM_LAYERS: int = 2
    N_INPUT_FEATURES: int = 2 # 3D input
    N_FUTURE_STEPS: int = 10
    SEQUENCE_LENGTH: int = 50
    TEACHER_FORCING: bool = True
    TEACHER_FORCING_RATIO: float = 0.7

    # simulator parameters
    dt = 20 # seconds. Time step
    T = 60 * 60 # seconds. Period of the orbit
    n = 2 * torch.pi / T # rad/s. Orbital rate
    N_TRAJ = 300 # Number of trajectories
    max_t = T # seconds. Maximum time to simulate