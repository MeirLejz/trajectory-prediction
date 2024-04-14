import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, 
                                out_features=output_size)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :]).unsqueeze(2)

        # self.fc = nn.Linear(hidden_size*(1+int(bidirectional)), output_length*num_of_output_features)
        # output = output[:, -1, :]
        # output = self.fc(output)
        # output = output.view(-1, self.OUTPUT_LENGTH, self.NUM_OUT_FEATURES)
        
        return out
    
