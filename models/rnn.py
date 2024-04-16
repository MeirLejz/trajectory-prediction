import torch.nn as nn
import torch

from ml_pipeline.hyperparams import Hyperparameters as hp
import pdb
class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        # self.linear = nn.Linear(in_features=hidden_size, 
        #                         out_features=output_size)
        
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=hp.N_FUTURE_STEPS*hp.N_INPUT_FEATURES)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        output, _ = self.lstm(x, (h0, c0))
        output = output[:, -1, :]
        output = self.linear(output)
        output = output.view(-1, hp.N_FUTURE_STEPS, hp.N_INPUT_FEATURES)

        # self.fc = nn.Linear(hidden_size*(1+int(bidirectional)), output_length*num_of_output_features)
        # output = self.fc(output)
        # output = output.view(-1, self.OUTPUT_LENGTH, self.NUM_OUT_FEATURES)

        return output
    
