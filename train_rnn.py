import matplotlib.pyplot as plt

import argparse, time

from torch import nn, torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

# from sklearn.preprocessing import MinMaxScaler

from trainer import Trainer
from dataset import CWTrajDataset
from simulator import CWSimulator
from hyperparams import Hyperparameters as hp

from models.encod_decod_lstm import LSTM_seq2seq

# # define scaler
# scaler = MinMaxScaler()
# # fit scaler on the training dataset
# scaler.fit(X_train)
# # transform both datasets
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

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

# itera = iter(train_dataloader)
# X, y = next(itera)
# print(f'{X.shape}')
# plt.figure()
# plt.scatter(X[0,:,0], X[0,:,1])
# plt.scatter(y[0,:,0], y[0,:,1])
# plt.show()

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[INFO] Using device: {device}')
    return device

def save_history(history: dict[str, list[float]], train_loss: float, val_loss: float, train_acc: float, val_acc: float) -> dict[str, list[float]]:
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    return history

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-lr", "--learning_rate", type=float, default=hp.INIT_LR, help="learning rate")
    ap.add_argument("-bs", "--batch_size", type=int, default=hp.BATCH_SIZE, help="batch size")
    ap.add_argument("-e", "--epochs", type=int, default=hp.EPOCHS, help="number of epochs")
    ap.add_argument("-m", "--model", type=str, default='output/model.pth', help="path to output trained model")
    ap.add_argument("-p", "--plot", type=str, default='output/plot.png', help="path to output loss plot")
    ap.add_argument("-t", "--train", type=bool, default=True, help="train the model")
    args = vars(ap.parse_args())  
    
    device = get_device()

    simulator = CWSimulator(dt=hp.dt, max_t=hp.max_t, n=hp.n, N_TRAJ=hp.N_TRAJ, SEQUENCE_LENGTH=hp.SEQUENCE_LENGTH)
    trajectories = simulator.simulate_trajectories()

    train_split = int(trajectories.shape[0] * hp.TRAIN_SPLIT)
    training_data = CWTrajDataset(trajectories=trajectories[:train_split], sequence_len=simulator.SEQUENCE_LENGTH, n_input_features=hp.N_INPUT_FEATURES, future_len=hp.N_FUTURE_STEPS)
    val_data = CWTrajDataset(trajectories=trajectories[train_split:], sequence_len=simulator.SEQUENCE_LENGTH, n_input_features=hp.N_INPUT_FEATURES, future_len=hp.N_FUTURE_STEPS)

    train_dataloader = DataLoader(training_data, batch_size=hp.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=hp.BATCH_SIZE)

    model = LSTM_seq2seq(input_size=hp.N_INPUT_FEATURES, hidden_size=hp.HIDDEN_SIZE, num_layers=hp.NUM_LAYERS, target_len=hp.N_FUTURE_STEPS).to(device)
    # model = LSTM(input_size=hp.N_INPUT_FEATURES, hidden_size=hp.HIDDEN_SIZE, output_size=hp.N_FUTURE_STEPS, num_layers=hp.NUM_LAYERS).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=hp.INIT_LR) # 
    scheduler = MultiStepLR(optimizer, milestones=hp.MILESTONES, gamma=0.1)

    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)

    H = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Training procedure
    start = time.time()
    for t in range(args["epochs"]):

        print(f"Epoch {t+1}/{args["epochs"]}\n-------------------------------")

        train_loss, train_acc = trainer.train(dataloader=train_dataloader)
        val_loss, val_acc = trainer.validate(dataloader=val_dataloader)
        scheduler.step()
        save_history(H, train_loss, val_loss, train_acc, val_acc)

    end = time.time()
    print(f"Done, Training time: {end-start} s")

    plot_results(history=H, path=args["plot"])
    torch.save(obj=model, f=args["model"])

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
