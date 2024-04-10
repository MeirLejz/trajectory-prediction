# matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse, time, numpy as np

from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR

from dataset import CWTrajDataset
from simulator import CWSimulator
from hyperparams import Hyperparameters as hp
from models.rnn import LSTM
from models.encod_decod_lstm import LSTM_seq2seq

from torchmetrics.functional import r2_score


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
        epoch_train_loss, epoch_train_acc = 0,0  
        model.to(device)
        model.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X, y)
            loss = loss_fn(pred.to(device), y)
            acc = r2_score_acc(pred.to(device), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()

            # print loss every 100 batches
            if batch % 100 == 0:
                loss, current = loss.item(), (batch+1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        
        epoch_train_loss /= num_batches
        epoch_train_acc /= num_batches
        print(f"Training Error: \n Avg loss: {epoch_train_loss:>8f}\nAvg R2 score (acc): {epoch_train_acc:>8f}")
        return epoch_train_loss, epoch_train_acc

    @staticmethod
    def validate(dataloader: DataLoader, model: nn.Module, loss_fn, device: torch.device) -> tuple[float, float]:
        
        _, num_batches = len(dataloader.dataset), len(dataloader)
        val_loss, val_acc = 0, 0
        model.to(device)
    
        model.eval()
        
        with torch.no_grad():
            
            for (X, y) in dataloader:

                X, y = X.to(device), y.to(device) # send input to device
                pred = model(X, y) # forward pass

                loss = loss_fn(pred.to(device), y) # compute loss
                acc = r2_score_acc(pred.to(device), y)

                val_loss += loss.item() # accumulate loss 
                val_acc += acc.item()

        val_loss /= num_batches
        val_acc /= num_batches
        
        print(f"Validation Error: \n Avg loss: {val_loss:>8f}\nAvg R2 score (acc): {val_acc:>8f}")
        return val_loss, val_acc
    
    @staticmethod
    def plot_results(history: dict, path: str) -> None:
        plt.style.use("ggplot")
        plt.figure()

        plt.plot([loss for loss in history["train_loss"]], label="train_loss")
        plt.plot([loss for loss in history["val_loss"]], label="val_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.savefig(path)

    @staticmethod
    def save_model(model: nn.Module, path: str) -> None:
        torch.save(obj=model, f=path)

def r2_score_acc(output: torch.Tensor, target: torch.Tensor):
    if len(output.shape) > 2:
        score = 0
        for i in range(output.shape[0]):
            score += r2_score(preds=output[i,:,:],target=target[i,:,:])
        return 1 - score / output.shape[0]
    else:
        return 1 - r2_score(preds=output,target=target)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-lr", "--learning_rate", type=float, default=hp.INIT_LR, help="learning rate")
    ap.add_argument("-bs", "--batch_size", type=int, default=hp.BATCH_SIZE, help="batch size")
    ap.add_argument("-e", "--epochs", type=int, default=hp.EPOCHS, help="number of epochs")
    ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained model")
    ap.add_argument("-p", "--plot", type=str, required=True, help="path to output loss plot")
    args = vars(ap.parse_args())  
    
    # looking for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[INFO] Using device: {device}')

    # initializing trainer object
    trainer = Trainer()

    # [INFO] Simulating trajectories...
    simulator = CWSimulator(dt=hp.dt, max_t=hp.max_t, n=hp.n, N_TRAJ=hp.N_TRAJ, SEQUENCE_LENGTH=hp.SEQUENCE_LENGTH)
    print(f'[INFO] Simulating {hp.N_TRAJ} trajectories...')
    trajectories = simulator.simulate_trajectories()
    trajectories = torch.tensor(trajectories).float()

    # if 2D, add a third dimension
    if len(trajectories.shape) == 2:
        trajectories = trajectories.unsqueeze(2)
    print(f'trajectories shape: {trajectories.shape}')

    print(f'[INFO] Creating datasets...')
    print("[INFO] generating the train/validation split...")
    train_split = int(trajectories.shape[0] * hp.TRAIN_SPLIT)
    training_data = CWTrajDataset(trajectories=trajectories[:train_split], sequence_len=simulator.SEQUENCE_LENGTH, n_input_features=hp.N_INPUT_FEATURES, future_len=hp.N_FUTURE_STEPS)
    val_data = CWTrajDataset(trajectories=trajectories[train_split:], sequence_len=simulator.SEQUENCE_LENGTH, n_input_features=hp.N_INPUT_FEATURES, future_len=hp.N_FUTURE_STEPS)

    print(f'[INFO] Training dataset size: {len(training_data)}')
    print(f'[INFO] Validation dataset size: {len(val_data)}')

    train_dataloader = DataLoader(training_data, batch_size=hp.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=hp.BATCH_SIZE)

    itera = iter(train_dataloader)
    X, y = next(itera)
    print(f'{X.shape}')
    plt.figure()
    plt.scatter(X[0,:,0], X[0,:,1])
    plt.scatter(y[0,:,0], y[0,:,1])
    plt.show()

    # model, loss function and optimization strategy definition
    model = LSTM_seq2seq(input_size=hp.N_INPUT_FEATURES, hidden_size=hp.HIDDEN_SIZE, num_layers=hp.NUM_LAYERS, target_len=hp.N_FUTURE_STEPS).to(device)
    # model = LSTM(input_size=hp.N_INPUT_FEATURES, hidden_size=hp.HIDDEN_SIZE, output_size=hp.N_FUTURE_STEPS, num_layers=hp.NUM_LAYERS).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=hp.INIT_LR) # 
    scheduler = MultiStepLR(optimizer, milestones=hp.MILESTONES, gamma=0.1)

    H = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    start = time.time()

    for t in range(args["epochs"]):

        print(f"Epoch {t+1}/{args["epochs"]}\n-------------------------------")
        train_loss, train_acc = trainer.train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, val_acc = trainer.validate(val_dataloader, model, loss_fn, device)
        scheduler.step()
        # print(f'Learning rate: {scheduler.get_last_lr()}')

        H["train_loss"].append(train_loss)
        H["val_loss"].append(val_loss)
        H["train_acc"].append(train_acc)
        H["val_acc"].append(val_acc)
        
    trainer.plot_results(history=H, path=args["plot"])
    trainer.save_model(model=model, path=args["model"])

    end = time.time()
    print(f"Done, Training time: {end-start} s")

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
    # usage:
    # python train_rnn.py -m output/model.pth -p output/plot.png

