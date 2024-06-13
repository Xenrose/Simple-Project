# modeling package
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ETC
import matplotlib.pyplot as plt
import numpy as np


torch.manual_seed(1234)

class DualInputLSTM(nn.Module):
    def __init__(self, device, params):
        super().__init__()
        self.history = None
        self.params = params
        self.stock_lstm_1 = nn.LSTM(input_size=params['stock_input'],
                                    hidden_size=params['stock_node'],
                                    batch_first=True,
                                    dropout=0.5,
                                    device=device)
        self.stock_lstm_2 = nn.LSTM(input_size=params['stock_node'],
                                    hidden_size=params['output_node'],
                                    batch_first=True,
                                    dropout=0.5,
                                    device=device)
            

        self.index_lstm_1 = nn.LSTM(input_size=params['index_input'],
                                    hidden_size=params['index_node'],
                                    batch_first=True,
                                    dropout=0.5,
                                    device=device)
        self.index_lstm_2 = nn.LSTM(input_size=params['index_node'],
                                    hidden_size=params['output_node'],
                                    batch_first=True,
                                    dropout=0.5,
                                    device=device)        

        self.dnese_layer = nn.Sequential(\
            nn.Linear(in_features=params['output_node'] * 7 * 2, 
                      out_features=self.params['dense_node']), 
            nn.Linear(in_features=self.params['dense_node'], 
                      out_features=1))
        

    def forward(self, stock, index):
        stock_output, _ = self.stock_lstm_1(stock)
        stock_output, _ = self.stock_lstm_2(stock_output)

        index_output, _ = self.index_lstm_1(index)
        index_output, _ = self.index_lstm_2(index_output)

        dense_input = torch.concat([stock_output, index_output]).view(stock.size(0), -1)

        dense_output = self.dnese_layer(dense_input)

        return dense_output    
    

    def set_history(self, history):
        self.history = history
        
    
    def get_history(self):
        if self.history is None:
            print("모델이 학습되지 않았습니다.")
        else:
            return self.history



def train(model, data, params, verbose:bool = True):    
    # data-loader
    train_loader = DataLoader(data['train'], batch_size=params['batch_size'], shuffle=False)
    valid_loader = DataLoader(data['valid'], batch_size=params['batch_size'], shuffle=False)
    
    train_loss_history = []
    valid_loss_history = []
    
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    loss_function = nn.MSELoss()


    for i in range(params['epoch']):
        _train_loss = 0
        model.train()
        with torch.enable_grad():
            for stock, index, label in train_loader:
                optimizer.zero_grad()
                outputs = model(stock, index).view(-1, 1)
                loss = loss_function(outputs, label)
                
                loss.backward()
                optimizer.step()

                _train_loss += loss.item()

        _train_loss /= params['batch_size']
        train_loss_history.append(_train_loss)


        _valid_loss = 0
        model.eval()
        with torch.no_grad():
            for stock, index, label in valid_loader:
                outputs = model(stock, index).view(-1, 1)
                loss = loss_function(outputs, label)
                
                _valid_loss += loss.item()

        _valid_loss /= params['batch_size']
        valid_loss_history.append(_valid_loss)

        if (i+1)%int((params['epoch'])*0.1)==0 and verbose:
            print(f"epoch : {i+1} Loss(train) : {train_loss_history[-1]:.3f}  Loss(valid) : {valid_loss_history[-1]:.3f}")

    if verbose:
        plt.plot(train_loss_history, label='Training loss')
        plt.plot(valid_loss_history, label='Validation loss')
        plt.legend()
        plt.show()
        print("="*40)
        print("Training lmmoss: ", train_loss_history[-1])
        print("Validation loss: ", valid_loss_history[-1])
        print("="*40)

    history = {"Train_loss": train_loss_history,
               "Valid_loss": valid_loss_history}
    model.set_history(history)



def predict(model, data, batch_size):
    test_loader = DataLoader(data['test'], batch_size=batch_size, shuffle=False)

    y_pred = torch.Tensor()
    y_test = torch.Tensor()

    model.eval()
    with torch.no_grad():
        for stock, index, label in test_loader:
            outputs = model(stock, index)
            y_pred = torch.concat([y_pred, outputs.cpu().view(-1)], dim=0)
            y_test = torch.concat([y_test, label.cpu().view(-1)], dim=0)

    
    return y_test, y_pred
