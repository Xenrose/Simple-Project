# modeling package
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
import numpy as np
import os
from pathlib import Path

tf.keras.utils.set_random_seed(1111)

class DualLSTM(tf.keras.models.Model):
    def __init__(self, **params):
        super().__init__()
        self.epoch = params['epoch']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        
        self.stock_node1 = params['stock_node1']
        self.stock_node2 = params['stock_node2']
        self.index_node1 = params['index_node1']
        self.index_node2 = params['index_node2']
        
        self.drop_out_rate = params['drop_out_rate']
        self.seed = params['seed']
        self.verbose = 0

        self.model = None
        self.history = None


    def create_model(self, stock_shape, y_shape, index_shape):
        input_stock = Input(shape=(stock_shape[1], stock_shape[2]))
        lstm_stock1 = LSTM(self.stock_node1, return_sequences=True, seed=self.seed)(input_stock)
        drop_1_1 = Dropout(self.drop_out_rate, seed=self.seed)(lstm_stock1)
        lstm_stock2 = LSTM(self.stock_node2, return_sequences=False, seed=self.seed)(drop_1_1)
        drop_1_2 = Dropout(self.drop_out_rate, seed=self.seed)(lstm_stock2)


        input_index = Input(shape=(index_shape[1], index_shape[2]))
        lstm_index1 = LSTM(self.index_node1, return_sequences=True, seed=self.seed)(input_index)
        drop_2_1 = Dropout(self.drop_out_rate, seed=self.seed)(lstm_index1)
        lstm_index2 = LSTM(self.index_node2, return_sequences=False, seed=self.seed)(drop_2_1)
        drop_2_2 = Dropout(self.drop_out_rate, seed=self.seed)(lstm_index2)


        concat = Concatenate()([drop_1_2, drop_2_2])
        output = Dense(y_shape[1])(concat)
        self.model = Model(inputs=[input_stock, input_index], outputs=output)


    def summary(self) -> None:
        if self.model is None:
            print("model이 생성되지 않았습니다.")
        else:
            print(self.model.summary())


    def model_export(self):
        if self.model is None:
            print("model이 생성되지 않았습니다.")
        else:
            return self.model
        

    def history_export(self):
        if self.history is None:
            print("학습이 진행되지 않았습니다.")
        else:
            return self.history.history
        

    def fit(self, stock:dict, index:dict) -> None:
        self.create_model(stock['X_train'].shape, stock['y_train'].shape, index['X_train'].shape)

        # hyperparameters
        self.model.compile(optimizer = Adam(learning_rate=self.learning_rate), 
                           loss='mse',
                           metrics=['mse'])

        early_stopping = EarlyStopping(
            monitor='val_loss',    
            patience=50,           
            verbose=0,             
            restore_best_weights=True)

        self.history = self.model.fit([stock['X_train'], index['X_train']], stock['y_train'], 
                                      epochs=self.epoch, batch_size=self.batch_size,
                                      validation_split=0.3, 
                                      verbose=self.verbose, 
                                      callbacks=[early_stopping])
        

    def predict(self, stock:dict, index:dict) -> np.array:
        pred = self.model.predict((stock['X_test'], index['X_test']), verbose = self.verbose)
        return pred
    
    
    def save(self) -> None:
        self.model.save(Path(os.getcwd(), 'lstm_keras.h5'))
