# max_temp_model.py
import numpy as np
import datetime
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from meteostat import Point, Daily

model_max = None
window_size = 2
all_we_x_max = None

def init_model():
    global model_max
    model_max = Sequential()
    model_max.add(Dense(10, input_dim=window_size, activation="relu"))
    model_max.add(Dense(1))
    model_max.compile(loss='mse', optimizer='adam')

def learn_max():
    global model_max, window_size, all_we_x_max
    if model_max is None:
        init_model()
    
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime.today()
    place = Point(55.755820, 37.617633)
    data = Daily(place, start_date, end_date).fetch().reset_index()
    
    temperatures = data['tmax'].dropna().values
    if len(temperatures) < window_size + 1:
        return
    
    X, y = [], []
    for i in range(len(temperatures) - window_size):
        X.append(temperatures[i:i + window_size])
        y.append(temperatures[i + window_size])
    
    X_train = np.array(X)
    y_train = np.array(y)
    
    model_max.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    all_we_x_max = X_train[-1] if len(X_train) > 0 else None

def predict_max():
    global all_we_x_max
    if all_we_x_max is None:
        return None
    return model_max.predict(np.array([all_we_x_max]), verbose=0)[0][0]