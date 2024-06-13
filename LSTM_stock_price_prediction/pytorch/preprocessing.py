# data processing package
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

class StockData(Dataset):
    def __init__(self, 
                 mode:str,
                 device,
                 data_path_stock:str = "../data/AAPL.csv",
                 data_path_index_1:str = '../data/SP500.csv',
                 data_path_index_2:str = '../data/NASDAQ.csv',
                 data_path_index_3:str = '../data/DOLLAR.csv',
                 split_date:str = '2023-08-01',
                 window_size:int = 7,
                 future_step:int = 1,
                 valid_size:float = 0.2) -> None:
        super().__init__()
        self.device = device
        stock = pd.read_csv(data_path_stock)
        SP500 = pd.read_csv(data_path_index_1)
        NASDAQ = pd.read_csv(data_path_index_2).drop(columns=['Date'])
        DOLLAR = pd.read_csv(data_path_index_3).drop(columns=['Date'])
        index = pd.concat([SP500, NASDAQ, DOLLAR], axis=1)


        if mode=="train" or mode=="valid":
            start = 0 if mode=="train" else int(len(stock)*(1-valid_size))
            end = int(len(stock)*(1-valid_size)) if mode=="train" else len(stock)
        
            stock = stock[stock['Date'] < split_date][start:end]
            index = index[index['Date'] < split_date][start:end]

            X_stock = stock.drop(columns=['Date', 'Close'])
            X_index = index.drop(columns=['Date'])
            y = stock['Close']


        elif mode=="test":
            stock = stock[stock['Date'] >= split_date]
            index = index[index['Date'] >= split_date]

            X_stock = stock.drop(columns=['Date', 'Close'])
            X_index = index.drop(columns=['Date'])
            y = stock['Close']


        X_stock_seq, X_index_seq, y_seq = [], [], []
        _size = len(X_stock)

        for i in range(window_size, _size-future_step +1):
            X_stock_seq.append(X_stock[i - window_size:i])
            X_index_seq.append(X_index[i - window_size:i])
            y_seq.append(y[i + future_step - 1:i + future_step])

        self.X_stock = np.array(X_stock_seq)
        self.X_index = np.array(X_index_seq)
        self.y = np.array(y_seq)


    def __getitem__(self, idx) -> torch.Tensor:
        return torch.Tensor(self.X_stock[idx]).to(self.device), \
               torch.Tensor(self.X_index[idx]).to(self.device), \
               torch.Tensor(self.y[idx]).to(self.device)


    def __len__(self):
        return len(self.X_stock)
    
    



