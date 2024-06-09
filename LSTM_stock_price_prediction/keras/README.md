# 프로젝트 설명

* LSTM 모델을 통한 AAPL(apple inc.) 주가 예측 모델 입니다.

<br>

## 1. 특징
* keras를 통해 LSTM을 구현하였습니다.
* optuna를 통해 hyperparameter tuning을 진행하였습니다.


## 2. 파일(함수) 설명

* `main.py`: LSTM(keras ver.) main 파일
* `models.py`: LSTM model class
* `preprocessing.py`: data preprocessing
* `evaluate.py`: predict data evaluate
* `lstm_keras.h5`: LSTM(keras ver.) keras model save

## 3. evaluate
<img src="https://github.com/Xenrose/my_project/blob/main/LSTM_stock_price_prediction/keras/figure/loss_compare.png">
<img src="https://github.com/Xenrose/my_project/blob/main/LSTM_stock_price_prediction/keras/figure/visualize.png">
<img src="https://github.com/Xenrose/my_project/blob/main/LSTM_stock_price_prediction/keras/figure/visualize_zoom.png">