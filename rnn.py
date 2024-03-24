import torch.nn as nn
import torch
class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, 
                                out_features=output_size)
        # self.hidden_cell = (torch.zeros(num_layers,1,self.hidden_layer_size),
        #                     torch.zeros(num_layers,1,self.hidden_layer_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # print(f'h0 shape: {h0.shape}, c0 shape: {c0.shape}')
        out, _ = self.lstm(x, (h0, c0))
        # import pdb; pdb.set_trace()
        out = self.linear(out[:, -1, :]).unsqueeze(2)
        # pdb.set_trace()
        return out
    
