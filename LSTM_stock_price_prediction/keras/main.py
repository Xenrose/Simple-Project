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


def objectiveLR(trial:Trial, stock, index):
    params = {
        'epoch': trial.suggest_int('epoch', 50, 500),
        'seed': trial.suggest_categorical('seed', [42]),
        'verbose': trial.suggest_categorical('verbose', [0]),
        'batch_size': trial.suggest_categorical('batch_size', [32]),
        'drop_out_rate': trial.suggest_categorical('drop_out_rate', [0.5]),

        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1),
        'stock_node1': trial.suggest_categorical('stock_node1', [2**i for i in range(4, 10)]),
        'stock_node2': trial.suggest_categorical('stock_node2', [2**i for i in range(4, 10)]),
        'index_node1': trial.suggest_categorical('index_node1', [2**i for i in range(4, 10)]),
        'index_node2': trial.suggest_categorical('index_node2', [2**i for i in range(4, 10)])
    }

    model = DualLSTM(**params)
    model.fit(stock, index)
    
    score = Kfold_evaluate(model, stock, index, k=5)
    return score

    
    


if __name__=="__main__":
    ############### data load & preprocessing ###############
    # stock data
    AAPL = pd.read_csv('../data/AAPL.csv') 
    stock, scaler = stock_preprocessing(AAPL, window_size=7, future_step=1, split_date='2023-08-01')

    # index data
    SP500 = pd.read_csv('../data/SP500.csv')
    DOLLAR = pd.read_csv('../data/DOLLAR.csv')
    NASDAQ = pd.read_csv('../data/NASDAQ.csv')
    index = index_preprocessing(SP500, DOLLAR, NASDAQ)


    ############### Hyperparameter tuning with optuna ###############
    study_name = "study_keras_lstm"
    storage_name = f"sqlite:///keras_optuna/{study_name}.db"

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


    study.optimize(lambda trial : objectiveLR(trial, stock, index), 
                   n_trials=10)


    ############### model create & fit & evaluate ###############

    lstm = DualLSTM(**study.best_params)
    lstm.fit(stock, index)
    y_pred = lstm.predict(stock, index)

    # 학습비용 절감 목적으로 target 또한 scaling 해주었으나
    # 시각화를 위해 원래의 값으로 다시 unpack 해줍니다.
    y_pred, y_test = unpack_scaled(y_pred, stock['y_test'], scaler) 


    lstm.summary()
    lstm.save()

    print("="*20, "best parameter" ,"="*20)
    print(study.best_value)
    print("="*55)
    print("="*20, "best parameter" ,"="*20)
    print(study.best_params)
    print("="*55)



    if not os.path.isdir(Path(os.getcwd(), "figure")):
        os.makedirs(Path(os.getcwd(), "figure"))

    loss_compare(lstm.history_export())
    visualize(AAPL, y_pred, y_test)
    zoom_visualize(AAPL, y_pred, y_test)