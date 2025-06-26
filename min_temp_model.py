# min_temp_model.py
import numpy as np
import datetime
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from meteostat import Point, Daily

model_min = None
window_size = 2
all_we_x_min = None

def init_model():
    global model_min
    model_min = Sequential()
    model_min.add(Dense(10, input_dim=window_size, activation="relu"))
    model_min.add(Dense(1))
    model_min.compile(loss='mse', optimizer='adam')

def learn_min():
    global model_min, window_size, all_we_x_min
    if model_min is None:
        init_model()
    
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime.today()
    place = Point(55.755820, 37.617633)
    data = Daily(place, start_date, end_date).fetch().reset_index()
    
    temperatures = data['tmin'].dropna().values
    if len(temperatures) < window_size + 1:
        return
    
    X, y = [], []
    for i in range(len(temperatures) - window_size):
        X.append(temperatures[i:i + window_size])
        y.append(temperatures[i + window_size])
    
    X_train = np.array(X)
    y_train = np.array(y)
    
    model_min.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    all_we_x_min = X_train[-1] if len(X_train) > 0 else None

def predict_min():
    global all_we_x_min
    if all_we_x_min is None:
        return None
    return model_min.predict(np.array([all_we_x_min]), verbose=0)[0][0]