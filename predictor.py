from simulator import CWSimulator
import matplotlib.pyplot as plt
import torch
import numpy as np
import pdb
from ml_pipeline.hyperparams import Hyperparameters as hp
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_model(path: str = './output/model.pth') -> torch.nn.Module:
    return torch.load(path)
    

def load_scaler(path: str = 'output/scaler.gz') -> MinMaxScaler:
    return joblib.load(path)

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'[INFO] Using device: {device}')
    return device

N_TRAJ_TEST = 3

def main():
    
    device = get_device()

    # Load the model
    model = load_model(path='./output/model.pth')
    scaler = load_scaler(path='./output/scaler.gz')
    
    # prediction only
    model.eval()

    # simulate a test dataset from the same feature distribution
    simulator = CWSimulator(dt=hp.dt, max_t=hp.max_t, n=hp.n, N_TRAJ=N_TRAJ_TEST, SEQUENCE_LENGTH=hp.SEQUENCE_LENGTH)
    trajectories = simulator.simulate_trajectories()

    seq_length = simulator.SEQUENCE_LENGTH

    ynn = np.zeros((simulator.N_TRAJ, len(simulator.times), hp.N_INPUT_FEATURES))
    ynn[:, 0:seq_length, :] = trajectories[:, 0:seq_length, :]
    print(f'lenght of times: {len(simulator.times)}')
    
    with torch.no_grad():
        for j in range(simulator.N_TRAJ):
            for k in range(seq_length, len(simulator.times) - hp.N_FUTURE_STEPS, hp.N_FUTURE_STEPS):
                # rnn_input = torch.tensor(ynn[j, k - seq_length:k, :]).float().unsqueeze(0)

                rnn_input = torch.Tensor(trajectories[j, k - seq_length:k, :]).unsqueeze(0) # trajectories[...]
                rnn_input_scaled = torch.Tensor(scaler.transform(rnn_input.view(-1, hp.N_INPUT_FEATURES))).view(1,seq_length,hp.N_INPUT_FEATURES)
                output = model(rnn_input_scaled.to(device)) # model.predict() for seq2seq LSTM
                output_unscaled = torch.Tensor(scaler.inverse_transform(output.cpu().view(-1, hp.N_INPUT_FEATURES))).view(1,hp.N_FUTURE_STEPS,hp.N_INPUT_FEATURES)
                ynn[j, k:k+hp.N_FUTURE_STEPS, :] = output_unscaled.numpy()

    N_steps = int(hp.T/hp.dt)


    ax = plt.figure().add_subplot(projection=None if hp.N_INPUT_FEATURES == 1 else '3d')
    for j in range(simulator.N_TRAJ):

        if hp.N_INPUT_FEATURES == 1:
            
            x = trajectories[j,:,:].squeeze().numpy()
            ax.plot(simulator.times, x, linewidth=1,marker='o')
            ax.plot(x[0], color='r')
            x_ = ynn[j,:,:].squeeze()
            ax.plot(simulator.times, x_, linewidth=1, linestyle='--', marker='x')
            ax.plot(x_[seq_length], color='m')
            ax.plot(0,0, color='k')

        elif hp.N_INPUT_FEATURES == 2:
            x, y = trajectories[j,:,:].squeeze().T.numpy()
            ax.plot(x, y, linewidth=1,marker='o')
            ax.scatter(x[0], y[0], color='r')
            x_, y_ = ynn[j,:,:].squeeze().T
            ax.plot(x_, y_, linewidth=1, linestyle='--', marker='x')
            ax.scatter(x_[seq_length], y_[seq_length], color='m')
            ax.scatter(0,0, color='k')

        elif hp.N_INPUT_FEATURES == 3:
            x, y, z = trajectories[j,:,:].squeeze().T.numpy()
            ax.plot(x, y, z, linewidth=1) # [:N_steps]
            ax.scatter(x[0], y[0], z[0], color='r')
            x_, y_, z_ = ynn[j,:,:].squeeze().T
            ax.scatter(x_, y_, z_, linewidth=1, linestyle='--', marker='x')
            # ax.scatter(x_[seq_length], y_[seq_length], z_[seq_length], color='m')
            ax.scatter(0,0,0, color='k')

    plt.show()
    pdb.set_trace()
                
if __name__ == '__main__':
    main()
    # adapt forward function of LSTM seq to seq
    # scale down & back trajectories