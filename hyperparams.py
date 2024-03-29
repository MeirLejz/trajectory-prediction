from dataclasses import dataclass

@dataclass
class Hyperparameters:
    # define training hyperparameters
    INIT_LR: float = 1e-2
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    # define the train and val splits
    TRAIN_SPLIT: float = 0.75
    VAL_SPLIT: float = 1 - TRAIN_SPLIT
    HIDDEN_SIZE: int = 16
    MILESTONES = [10,50]
    NUM_LAYERS: int = 2

    N_INPUT_FEATURES: int = 2 # 3D input
    N_FUTURE_STEPS: int = 10
    SEQUENCE_LENGTH: int = 50