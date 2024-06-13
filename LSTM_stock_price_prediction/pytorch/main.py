# package import
from models import *
from evaluate import *
from preprocessing import *

import optuna 
from optuna import Trial 
from optuna.samplers import TPESampler 

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



def objectiveLR(trial:Trial, data, device):
    params = {
        # 고정값 
        'stock_input': trial.suggest_categorical('stock_input', [8]), # (7, 9)
        'index_input': trial.suggest_categorical('index_input', [27]), # (7, 10)
        'batch_size': trial.suggest_categorical('batch_size', [32]),

        'epoch': trial.suggest_int('epoch', 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1),
        'stock_node': trial.suggest_categorical('stock_node', [2**i for i in range(4, 10)]),
        'index_node': trial.suggest_categorical('index_node', [2**i for i in range(4, 10)]),
        'output_node': trial.suggest_categorical('output_node', [2**i for i in range(4, 10)]),

        'dense_node': trial.suggest_int('dense_node', 10, 100),
    }

    model = DualInputLSTM(device, params).to(device)
    train(model, data, params, verbose=True)
    
    return model.get_history()['Train_loss'][-1]





if __name__=="__main__":
    ############### data load & preprocessing ###############
    data = {"train": StockData(mode='train', device=device),
            "valid": StockData(mode='valid', device=device),
            "test": StockData(mode='test', device=device)}

    ############## Hyperparameter tuning with optuna ###############
    study_name = "study_pytorch_lstm"
    storage_name = f"sqlite:///pytorch_optuna/{study_name}.db"

    try:
        study = optuna.create_study(storage=storage_name,
                                    study_name=study_name,
                                    direction='minimize', 
                                    sampler=TPESampler(multivariate=True, n_startup_trials=50, seed=42))
        print("create")

    except:
        study = optuna.load_study(study_name=study_name, 
                                  storage=storage_name)
        print("load")


    study.optimize(lambda trial : objectiveLR(trial, data, device), n_trials=10)


    ############### model create & fit & evaluate ###############

    lstm = DualInputLSTM(device, study.best_params).to(device)
    train(lstm, data, study.best_params, verbose=True)
    y_test, y_pred = predict(lstm, data, study.best_params['batch_size'])


    print("="*20, "best parameter" ,"="*20)
    print(study.best_value)
    print("="*55)
    print("="*20, "best parameter" ,"="*20)
    print(study.best_params)
    print("="*55)



    if not os.path.isdir(Path(os.getcwd(), "figure")):
        os.makedirs(Path(os.getcwd(), "figure"))

    AAPL = pd.read_csv("../data/AAPL.csv")
    loss_compare(lstm.get_history())
    visualize(AAPL, y_pred)
    zoom_visualize(AAPL, y_pred)
