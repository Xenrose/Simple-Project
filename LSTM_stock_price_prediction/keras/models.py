# modeling package
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from evaluate import loss_compare
import tensorflow as tf


tf.keras.utils.set_random_seed(1111)


##############################################
######### stock + index + noise LSTM #########
##############################################
def triple_lstm_model(config, X_stock, y_stock, X_index, X_noise, model=False):
    if model:
        model = model
    else:
        # stock
        
        input_stock = Input(shape=(X_stock.shape[1], X_stock.shape[2]))
        lstm_stock1 = LSTM(config.lstm_node1, return_sequences=True, seed=config.seed)(input_stock)
        drop_1_1 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_stock1)
        lstm_stock2 = LSTM(config.lstm_node2, return_sequences=False, seed=config.seed)(drop_1_1)
        drop_1_2 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_stock2)


        # index
        input_index = Input(shape=(X_index.shape[1], X_index.shape[2]))
        lstm_index1 = LSTM(config.lstm_node1, return_sequences=True, seed=config.seed)(input_index)
        drop_2_1 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_index1)
        lstm_index2 = LSTM(config.lstm_node2, return_sequences=False, seed=config.seed)(drop_2_1)
        drop_2_2 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_index2)


        # noise
        input_noise = Input(shape=(X_noise.shape[1], X_noise.shape[2]))
        lstm_noise1 = LSTM(config.lstm_node1, return_sequences=True, seed=config.seed)(input_noise)
        drop_3_1 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_noise1)
        lstm_noise2 = LSTM(config.lstm_node2, return_sequences=False, seed=config.seed)(drop_3_1)
        drop_3_2 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_noise2)


        # output
        concat = Concatenate()([drop_1_2, drop_2_2, drop_3_2])

        output = Dense(y_stock.shape[1])(concat)

        model = Model(inputs=[input_stock, input_index, input_noise], outputs=output)
        model.summary()


    # hyperparameters
    optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='mse')

    early_stopping = EarlyStopping(
        monitor='val_loss',    
        patience=40,           
        verbose=0,             
        restore_best_weights=True 
    )
    
    history = model.fit([X_stock, X_index, X_noise], y_stock, epochs=config.epochs, batch_size=config.batch_size,
                validation_split=config.validation_split, verbose=config.verbose, callbacks=[early_stopping])


    loss_compare(history)
    return model



##############################################
######### stock + index or noise LSTM ########
##############################################
def dual_lstm_model(config, X_stock, y_stock, X_index, model=False):
    if model:
        model = model
    else:
        # stock
        input_stock = Input(shape=(X_stock.shape[1], X_stock.shape[2]))
        lstm_stock1 = LSTM(config.lstm_node1, return_sequences=True, seed=config.seed)(input_stock)
        drop_1_1 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_stock1)
        lstm_stock2 = LSTM(config.lstm_node2, return_sequences=False, seed=config.seed)(drop_1_1)
        drop_1_2 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_stock2)

        # index or noise
        input_index = Input(shape=(X_index.shape[1], X_index.shape[2]))
        lstm_index1 = LSTM(config.lstm_node1, return_sequences=True, seed=config.seed)(input_index)
        drop_2_1 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_index1)
        lstm_index2 = LSTM(config.lstm_node2, return_sequences=False, seed=config.seed)(drop_2_1)
        drop_2_2 = Dropout(config.drop_out_rate, seed=config.seed)(lstm_index2)

        # output
        concat = Concatenate()([drop_1_2, drop_2_2])

        output = Dense(y_stock.shape[1])(concat)

        model = Model(inputs=[input_stock, input_index], outputs=output)
        model.summary()


    # hyperparameters
    optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='mse')

    early_stopping = EarlyStopping(
        monitor='val_loss',    
        patience=40,           
        verbose=0,             
        restore_best_weights=True 
    )

    history = model.fit([X_stock, X_index], y_stock, epochs=config.epochs, batch_size=config.batch_size,
                validation_split=config.validation_split, verbose=config.verbose, callbacks=[early_stopping])


    loss_compare(history)
    return model



##############################################
################ stock LSTM ##################
##############################################
def lstm_model(config, X, y, model=False):
    if model:
        model = model
    else:
        # stock
        model = Sequential()
        model.add(LSTM(config.lstm_node1, input_shape=(X.shape[1], X.shape[2]), # (seq length, input dimension)
                return_sequences=True, seed=config.seed))
        model.add(Dropout(config.drop_out_rate, seed=config.seed))
        model.add(LSTM(config.lstm_node2, return_sequences=False, seed=config.seed))
        model.add(Dropout(config.drop_out_rate, seed=config.seed))
        model.add(Dense(y.shape[1]))

        model.summary()

    # hyperparameters
    optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='mse')

    early_stopping = EarlyStopping(
        monitor='val_loss',    
        patience=40,           
        verbose=0,             
        restore_best_weights=True 
    )
    
    history = model.fit(X, y, epochs=config.epochs, batch_size=config.batch_size,
                validation_split=config.validation_split, verbose=config.verbose, callbacks=[early_stopping])

    loss_compare(history)
    return model
