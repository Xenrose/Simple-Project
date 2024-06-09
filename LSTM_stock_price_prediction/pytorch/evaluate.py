# data processing package
import pandas as pd
import matplotlib.pyplot as plt


# evaluation package
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error



##############################################
################ loss_compare ################
##############################################
def loss_compare(history):
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()
    print("="*30)
    print("Training loss: ", history.history['loss'][-1])
    print("Validation loss: ", history.history['val_loss'][-1])
    print("="*30)




##############################################
############### predict_compare ##############
##############################################
def visualize(data, y_pred, y_test, file_name = False):
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
    plt.title(f'Original, {file_name} Close Price')
    plt.legend()

    if file_name:
        plt.savefig(f"../figure/{file_name}.png")

    plt.show()
    


##############################################
############ predict_compare_zoom ############
##############################################
def zoom_visualize(data, y_pred, y_test, file_name = False):
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
    plt.title(f'Zoomed In {file_name} Close Price')
    plt.legend()

    if file_name:
        plt.savefig(f"../figure/{file_name}_zoom.png")

    plt.show()

    

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