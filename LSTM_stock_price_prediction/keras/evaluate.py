# data processing package
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# evaluation package
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold


##############################################
################ loss_compare ################
##############################################
def loss_compare(history):
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.legend()
    # plt.show()
    plt.savefig(Path(os.getcwd(), "figure", "loss_compare.png"))


    print("="*30)
    print("Training loss: ", history['loss'][-1])
    print("Validation loss: ", history['val_loss'][-1])
    print("="*30)




##############################################
############### predict_compare ##############
##############################################
def visualize(data, y_pred, y_test):
    date = pd.to_datetime(data['Date'])

    # plotting
    plt.figure(figsize=(14, 7))

    # plot original 'Open' prices
    plt.plot(date, data['Close'], color='green', label='Original Close Price')

    temp_size = len(y_pred)
    # plot actual vs predicted
    plt.plot(date[len(data) - temp_size:], y_test, color='blue', label='Actual Close Price')
    plt.plot(date[len(data) - temp_size:], y_pred, color='red', linestyle='--', label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Original, Close Price')
    plt.legend()
    # plt.show()

    plt.savefig(Path(os.getcwd(), "figure", "visualize.png"))
    


##############################################
############ predict_compare_zoom ############
##############################################
def zoom_visualize(data, y_pred, y_test):
    date = pd.to_datetime(data['Date'])
    
    zoom_start = len(data) - len(y_pred)
    zoom_end = len(data)

    
    plt.figure(figsize=(14, 7))
    plt.plot(date[zoom_start:zoom_end],
            y_test,
            color='blue',
            label='Actual Close Price')

    plt.plot(date[zoom_start:zoom_end],
            y_pred,
            color='red',
            linestyle='--',
            label='Predicted Close Price')

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Zoomed In Close Price')
    plt.legend()
        
    plt.savefig(Path(os.getcwd(), "figure", "visualize_zoom.png"))

    # plt.show()

    

##############################################
############### MAE, MAPE report #############
##############################################
def evaluate_report(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("="*50)
    print("Mean Squared Error: ", mse.round(3))
    print("Mean Absolute Percentage Error: ", mape.round(3))
    print("="*50)
    
    return mse, mape


##############################################
##### Kfolad validation score with optuna #####
##############################################
def Kfold_evaluate(model, stock, index, k:int = 5) -> float:
    kf = KFold(n_splits=k, shuffle=False)
    scores = []
    stock_X = stock['X_train']
    index_X = index['X_train']
    y = stock['y_train']


    
    for train_idx, test_idx in kf.split(stock_X):
        stock = {}
        index = {}
        stock['X_train'], stock['X_test'] = stock_X[train_idx], stock_X[test_idx]
        index['X_train'], index['X_test'] = index_X[train_idx], index_X[test_idx]
        stock['y_train'], stock['y_test'] = y[train_idx], y[test_idx]


        y_pred = model.predict(stock, index)
        score = mean_squared_error(stock['y_test'], y_pred)
        scores.append(score)

    return sum(scores)/len(scores)