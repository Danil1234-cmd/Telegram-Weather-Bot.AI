# avg_temp_model.py
import numpy as np
import datetime
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from meteostat import Point, Daily

model_avg = None
window_size = 2
all_we_x_avg = None

def init_model():
    global model_avg
    model_avg = Sequential()
    model_avg.add(Dense(10, input_dim=window_size, activation="relu"))
    model_avg.add(Dense(1))
    model_avg.compile(loss='mse', optimizer='adam')

def learn_avg():
    global model_avg, window_size, all_we_x_avg
    if model_avg is None:
        init_model()
    
    start_date = datetime.datetime(2024, 1, 1)
    end_date = datetime.datetime.today()
    place = Point(55.755820, 37.617633)
    data = Daily(place, start_date, end_date).fetch().reset_index()
    
    temperatures = data['tavg'].dropna().values
    if len(temperatures) < window_size + 1:
        return
    
    X, y = [], []
    for i in range(len(temperatures) - window_size):
        X.append(temperatures[i:i + window_size])
        y.append(temperatures[i + window_size])
    
    X_train = np.array(X)
    y_train = np.array(y)
    
    model_avg.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    all_we_x_avg = X_train[-1] if len(X_train) > 0 else None

def predict_avg():
    global all_we_x_avg
    if all_we_x_avg is None:
        return None
    return model_avg.predict(np.array([all_we_x_avg]), verbose=0)[0][0]