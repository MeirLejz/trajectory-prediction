from torch import torch, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.functional import r2_score
import pdb
class Trainer():

    def __init__(self, model: nn.Module, loss_fn, optimizer: Optimizer, device: torch.device) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
    def train(self, dataloader: DataLoader) -> tuple[float, float]:
        
        size, num_batches = len(dataloader.dataset), len(dataloader)
        epoch_train_loss, epoch_train_acc = 0, 0  
        
        self.model.to(self.device)
        self.model.train()

        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(self.device), y.to(self.device)

            # pred = self.model(x=X, target=y).to(self.device)
            pred = self.model(x=X).to(self.device)

            loss = self.loss_fn(input=pred, target=y)
            acc = self.r2_score_acc(input=pred, target=y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

    def validate(self, dataloader: DataLoader) -> tuple[float, float]:
        
        _, num_batches = len(dataloader.dataset), len(dataloader)
        val_loss, val_acc = 0, 0
        
        self.model.to(self.device) 
        self.model.eval()
        
        with torch.no_grad():
            
            for (X, y) in dataloader:

                X, y = X.to(self.device), y.to(self.device) # send input to device
                
                # pred = self.model(x=X, target=y).to(self.device) # forward pass
                pred = self.model(x=X).to(self.device)

                loss = self.loss_fn(input=pred, target=y) # compute loss
                acc = self.r2_score_acc(input=pred, target=y)

                val_loss += loss.item() # accumulate loss 
                val_acc += acc.item()

        val_loss /= num_batches
        val_acc /= num_batches
        
        print(f"Validation Error: \n Avg loss: {val_loss:>8f}\nAvg R2 score (acc): {val_acc:>8f}")
        return val_loss, val_acc
    
    @staticmethod
    def r2_score_acc(input: torch.Tensor, target: torch.Tensor):
        if len(input.shape) > 2:
            score = 0
            for i in range(input.shape[0]):
                score += r2_score(preds=input[i,:,:],target=target[i,:,:])
            return score / input.shape[0]
        else:
            return r2_score(preds=input,target=target)

