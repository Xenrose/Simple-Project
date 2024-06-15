# data processing package
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 학습 비용 절감을 위해 target 또한 scaling을 해줍니다. 
def stock_preprocessing(stock_data, split_date='2023-08-01', window_size=7, future_step=1):
    train = stock_data[stock_data['Date'] < split_date].drop(['Date'], axis=1)
    test = stock_data[stock_data['Date'] >= split_date].drop(['Date'], axis=1)
    


    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)    
    
    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=test.columns)

    
    X_train = train_scaled.drop(columns=['Close', 'Adj Close']).to_numpy()
    y_train = train_scaled['Close'].to_numpy()
    X_test = test_scaled.drop(columns=['Close', 'Adj Close']).to_numpy()
    y_test = test_scaled['Close'].to_numpy()


    X_seq_train, y_seq_train = [], []
    X_seq_test, y_seq_test = [], []

    train_size = len(y_train)
    test_size = len(y_test)

    for i in range(window_size, train_size-future_step +1):
        X_seq_train.append(X_train[i - window_size:i])
        y_seq_train.append(y_train[i + future_step - 1])

    for i in range(window_size, test_size-future_step +1):
        X_seq_test.append(X_test[i - window_size:i])
        y_seq_test.append(y_test[i + future_step - 1])


    stock = {"X_train": np.array(X_seq_train),
             "y_train": np.array(y_seq_train),
             "X_test": np.array(X_seq_test),
             "y_test": np.array(y_seq_test),}
    
    return stock, scaler



def index_preprocessing(SP500, DOLLAR, NASDAQ):
    X_seq_train_sp500, X_seq_test_sp500 = preprocessing_index_noise(SP500)
    X_seq_train_dollar, X_seq_test_dollar = preprocessing_index_noise(DOLLAR)
    X_seq_train_nasdaq, X_seq_test_nasdaq = preprocessing_index_noise(NASDAQ)
    

    X_seq_train_index = np.concatenate((X_seq_train_sp500, X_seq_train_dollar, X_seq_train_nasdaq), axis=1)
    X_seq_test_index = np.concatenate((X_seq_test_sp500, X_seq_test_dollar, X_seq_test_nasdaq), axis=1)


    index = {"X_train": X_seq_train_index,
             "X_test": X_seq_test_index}
    
    return index


def preprocessing_index_noise(data, split_date='2023-08-01', window_size=7, future_step=1):
    train = data[data['Date'] < split_date].drop(['Date'], axis=1)
    test = data[data['Date'] >= split_date].drop(['Date'], axis=1)

    train_size = len(train)
    test_size = len(test)

    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    X_seq_train, X_seq_test = [], []


    for i in range(window_size, train_size-future_step +1):
        X_seq_train.append(train[i - window_size:i])

    for i in range(window_size, test_size-future_step +1):
        X_seq_test.append(test[i - window_size:i])   

    return np.array(X_seq_train), np.array(X_seq_test)



def unpack_scaled(y_pred, y_test, scaler):
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], y_pred.shape[0], axis=0)
    mean_values_pred[:, 0] = np.squeeze(y_pred)
    y_pred = scaler.inverse_transform(mean_values_pred)[:,0]

    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], y_test.shape[0], axis=0)
    mean_values_testY[:, 0] = np.squeeze(y_test)
    y_test = scaler.inverse_transform(mean_values_testY)[:,0]
    
    return y_pred, y_test