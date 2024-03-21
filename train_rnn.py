import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse, time, numpy as np

from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision import datasets
from torch.optim import Optimizer, Adam
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR


from dataset import CWTrajDataset
from simulator import CWSimulator
from hyperparams import Hyperparameters as hp
from rnn import LSTM

# test_data = datasets.KMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=transform,
# )

class Trainer():

    def __init__(self):
        pass

    @staticmethod
    def get_mean_std(loader: DataLoader) -> tuple[float, float]:
        iterator = iter(loader)
        batch, _ = next(iterator)
        
        mean, std = np.zeros((batch.shape[1])), np.zeros((batch.shape[1]))

        for image_batch, _ in loader:

            mean += image_batch.mean(dim=(0,2,3)).numpy()
            std += image_batch.std(dim=(0,2,3)).numpy()
        
        mean = np.divide(mean, len(loader))
        std = np.divide(std, len(loader))
        return (mean.tolist(), std.tolist())

    @staticmethod
    def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: Optimizer, device: torch.device) -> float:
        
        size, num_batches = len(dataloader.dataset), len(dataloader)
        epoch_train_loss = 0  

        model.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            # print loss every 100 batches
            if batch % 100 == 0:
                loss, current = loss.item(), (batch+1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        
        epoch_train_loss /= num_batches
        print(f"Training Error: \n Avg loss: {epoch_train_loss:>8f}\n")
        return epoch_train_loss

    @staticmethod
    def validate(dataloader: DataLoader, model: nn.Module, loss_fn, device: torch.device) -> tuple[float, float]:
        
        size, num_batches = len(dataloader.dataset), len(dataloader)
        val_loss = 0
    
        model.eval()
        
        with torch.no_grad():
            
            for (X, y) in dataloader:

                X, y = X.to(device), y.to(device) # send input to device
                pred = model(X) # forward pass

                loss = loss_fn(pred, y) # compute loss

                val_loss += loss.item() # accumulate loss 
        
        val_loss /= num_batches
        
        print(f"Validation Error: \n Avg loss: {val_loss:>8f}\n")
        return val_loss
    
    @staticmethod
    def plot_results(history: dict, path: str) -> None:
        plt.style.use("ggplot")
        plt.figure()

        plt.plot([loss for loss in history["train_loss"] if loss < 1], label="train_loss")
        plt.plot([loss for loss in history["val_loss"] if loss < 1], label="val_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(path)

    @staticmethod
    def save_model(model: nn.Module, path: str) -> None:
        torch.save(obj=model, f=path)

def main():

    N_INPUT_FEATURES = 1 # 1D

    ap = argparse.ArgumentParser()
    ap.add_argument("-lr", "--learning_rate", type=float, default=hp.INIT_LR, help="learning rate")
    ap.add_argument("-bs", "--batch_size", type=int, default=hp.BATCH_SIZE, help="batch size")
    ap.add_argument("-e", "--epochs", type=int, default=hp.EPOCHS, help="number of epochs")
    ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained model")
    ap.add_argument("-p", "--plot", type=str, required=True, help="path to output loss plot")
    args = vars(ap.parse_args())  
    
    # looking for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initializing trainer object
    trainer = Trainer()

    # [INFO] Simulating trajectories...
    simulator = CWSimulator()
    trajectories = simulator.simulate_trajectories()
    trajectories = torch.tensor(trajectories).float()


    # if 2D, add a third dimension
    if len(trajectories.shape) == 2:
        trajectories = trajectories.unsqueeze(2)
    print(f'trajectories shape: {trajectories.shape}')

    print(f'[INFO] Creating datasets...')
    print("[INFO] generating the train/validation split...")
    train_split = int(trajectories.shape[0] * hp.TRAIN_SPLIT)
    training_data = CWTrajDataset(trajectories=trajectories[:train_split], sequence_len=simulator.SEQUENCE_LENGTH, n_input_features=N_INPUT_FEATURES)
    val_data = CWTrajDataset(trajectories=trajectories[train_split:], sequence_len=simulator.SEQUENCE_LENGTH, n_input_features=N_INPUT_FEATURES)

    # print(f'Length of training data: {len(training_data)}')
    # training_data_size = int(len(training_data) * hp.TRAIN_SPLIT)
    # val_data_size = len(training_data) - training_data_size # int(len(training_data) * hp.VAL_SPLIT)
    # (training_data, val_data) = random_split(training_data, [training_data_size, val_data_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(training_data, batch_size=hp.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=hp.BATCH_SIZE)
    
    # print(f'dataloader length: {len(train_dataloader)}')

    # batch, truth = next(iter(train_dataloader)) 
    # print(f'batch length: {len(batch)}')
    # fig, ax = plt.subplots(1,1)
    # for i in range(batch.shape[0]):

    #     x = batch[i, :, :].squeeze().numpy().T
        
    #     ax.plot(x, marker='o', linewidth=1)
    #     ax.scatter(0, x[0], color='r')
    #     x_pred = truth[i, :].squeeze().numpy()
    #     ax.scatter(10, x_pred, color='m')
    # plt.show()
    # import pdb; pdb.set_trace()

    # model, loss function and optimization strategy definition
    model = LSTM(input_size=N_INPUT_FEATURES, hidden_size=hp.HIDDEN_SIZE, output_size=N_INPUT_FEATURES, num_layers=hp.NUM_LAYERS).to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=hp.INIT_LR) # 
    # scheduler = MultiStepLR(optimizer, milestones=[200,600, 800], gamma=0.1)

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    start = time.time()

    for t in range(args["epochs"]):

        print(f"Epoch {t+1}/{args["epochs"]}\n-------------------------------")
        train_loss = trainer.train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss = trainer.validate(val_dataloader, model, loss_fn, device)
        # scheduler.step()
        # print(f'Learning rate: {scheduler.get_last_lr()}')

        H["train_loss"].append(train_loss)
        H["val_loss"].append(val_loss)
        
    trainer.plot_results(history=H, path=args["plot"])
    trainer.save_model(model=model, path=args["model"])

    end = time.time()
    print(f"Done, Training time: {end-start} s")

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()

