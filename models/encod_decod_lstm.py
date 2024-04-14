import torch.nn as nn
import torch
import random
from ml_pipeline.hyperparams import Hyperparameters as hp
class LSTM_Encoder(nn.Module):
    
    """ Encoder module for seq2seq LSTM. Encodes input sequence into hidden states. """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):

        super(LSTM_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, (hidden, cell) = self.lstm(x, (h0, c0))

        return out, (hidden, cell)

class LSTM_Decoder(nn.Module):
        
        """ Decoder module for seq2seq LSTM. Decodes hidden states into output sequence. """
    
        def __init__(self, input_size: int, hidden_size: int, num_layers: int):
    
            super(LSTM_Decoder, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size=self.input_size, 
                                hidden_size=self.hidden_size, 
                                num_layers=self.num_layers, 
                                batch_first=True)
            self.linear = nn.Linear(in_features=self.hidden_size, 
                                    out_features=self.input_size)
    
        def forward(self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            out, self.hidden = self.lstm(x.unsqueeze(1), hidden)
            out = self.linear(out.squeeze(1))    
            return out, self.hidden

class LSTM_seq2seq(nn.Module):
    
    """ seq2seq LSTM model. Combines Encoder and Decoder modules. """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, target_len: int, teacher_forcing: bool=hp.TEACHER_FORCING, teacher_forcing_ratio: float=hp.TEACHER_FORCING_RATIO):
        
        super(LSTM_seq2seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_len = target_len

        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.encoder = LSTM_Encoder(input_size=self.input_size, 
                                    hidden_size=self.hidden_size, 
                                    num_layers=self.num_layers)
        
        self.decoder = LSTM_Decoder(input_size=self.input_size, 
                                    hidden_size=self.hidden_size, 
                                    num_layers=self.num_layers)
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        outputs = torch.zeros(x.shape[0], self.target_len, x.shape[2])

        _, (hidden, cell) = self.encoder(x)
        decoder_input = x[:, -1, :]
        decoder_hidden = (hidden, cell)

        # recursive method to generate output sequence, no teacher forcing
        if self.teacher_forcing and random.random() < self.teacher_forcing_ratio:
            for t in range(self.target_len):
                decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_out
                decoder_input = target[:, t, :]
        else:
            for t in range(self.target_len):
                decoder_out, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_out
                decoder_input = decoder_out    

        return outputs

    def predict(self, x: torch.Tensor) -> torch.Tensor:

        encoder_output, encoder_hidden = self.encoder(x)

        outputs = torch.zeros(x.shape[0], self.target_len, x.shape[2])

        decoder_input = x[:,-1,:]
        decoder_hidden = encoder_hidden

        for t in range(self.target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            import pdb; pdb.set_trace()
            outputs[:,t,:] = decoder_output

        return outputs