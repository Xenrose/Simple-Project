# data processing package
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# evaluation package
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold

from models import *

##############################################
################ loss_compare ################
##############################################
def loss_compare(history):
    plt.plot(history['Train_loss'], label='Training loss')
    plt.plot(history['Valid_loss'], label='Validation loss')
    plt.legend()
    # plt.show()
    plt.savefig(Path(os.getcwd(), "figure", "loss_compare.png"))


    print("="*30)
    print("Training loss: ", history['Train_loss'][-1])
    print("Validation loss: ", history['Valid_loss'][-1])
    print("="*30)




##############################################
############### predict_compare ##############
##############################################
def visualize(stock, y_pred):
    stock['Date'] = pd.to_datetime(stock['Date'])

    # plotting
    plt.figure(figsize=(14, 7))
    
    # plot original 'Open' prices
    plt.plot(stock['Date'], stock['Close'], color='green', label='Original Close Price')

    target_df = stock.iloc[len(stock) - len(y_pred):, :]

    # plot actual vs predicted
    plt.plot(target_df['Date'], target_df['Close'], color='blue', label='Actual Close Price')
    plt.plot(target_df['Date'], y_pred, color='red', linestyle='--', label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Original, Close Price')
    plt.legend()
    # plt.show()

    plt.savefig(Path(os.getcwd(), "figure", "visualize.png"))


##############################################
############ predict_compare_zoom ############
##############################################
def zoom_visualize(stock, y_pred):
    date = pd.to_datetime(stock['Date'])
    
    zoom_start = len(stock) - len(y_pred)
    zoom_end = len(stock)

    
    plt.figure(figsize=(14, 7))
    plt.plot(date[zoom_start:zoom_end],
            stock.loc[zoom_start:zoom_end, 'Close'],
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

