# data processing package
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



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

    train_size = len(train)
    test_size = len(test)

    for i in range(window_size, train_size-future_step +1):
        X_seq_train.append(X_train[i - window_size:i])
        y_seq_train.append(y_train[i + future_step - 1:i + future_step])

    for i in range(window_size, test_size-future_step +1):
        X_seq_test.append(X_test[i - window_size:i])
        y_seq_test.append(y_test[i + future_step - 1:i + future_step])


    return np.array(X_seq_train), np.array(y_seq_train), np.array(X_seq_test), np.array(y_seq_test), scaler
    



def preprocessing_index_noise(data, data_type, split_date='2023-08-01', window_size=7, future_step=1):
    train = data[data['Date'] < split_date].drop(['Date'], axis=1)
    test = data[data['Date'] >= split_date].drop(['Date'], axis=1)

    train_size = len(train)
    test_size = len(test)


    if data_type == 'index':
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
    # generate array filled with means for prediction
    mean_values_pred = np.repeat(scaler.mean_[np.newaxis, :], y_pred.shape[0], axis=0)

    # substitute predictions into the first column
    mean_values_pred[:, 0] = np.squeeze(y_pred)

    # inverse transform
    y_pred = scaler.inverse_transform(mean_values_pred)[:,0]
    # print(y_pred.shape)

    # generate array filled with means for testY
    mean_values_testY = np.repeat(scaler.mean_[np.newaxis, :], y_test.shape[0], axis=0)

    # substitute testY into the first column
    mean_values_testY[:, 0] = np.squeeze(y_test)

    # inverse transform
    y_test = scaler.inverse_transform(mean_values_testY)[:,0]
    # print(testY_original.shape)
    
    return y_pred, y_test
