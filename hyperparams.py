from dataclasses import dataclass

@dataclass
class Hyperparameters:
    # define training hyperparameters
    INIT_LR: float = 1e-3
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    # define the train and val splits
    TRAIN_SPLIT: float = 0.75
    VAL_SPLIT: float = 1 - TRAIN_SPLIT
    HIDDEN_SIZE: int = 16
    NUM_LAYERS: int = 2
