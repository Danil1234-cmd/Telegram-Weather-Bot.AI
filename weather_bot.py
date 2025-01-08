'''Спасибо за использование репозитория. 
В коде будут коментарии, следуя которым Вы сможите настроить бот для использования
Вот команды для скачивания библиотек для работы бота: 
python3 -m pip install keras
python3 -m pip install tensorflow
python3 -m pip install numpy
python3 -m pip install datetime
python3 -m pip install meteoostat
python3 -m pip install telebot
python3 -m pip install schedule
python3 -m pip install time
python3 -m pip install threading  
'''



from keras import Sequential
from keras.layers import Dense
import numpy as np
import datetime
from meteostat import Point, Daily
import telebot
import schedule
import time
from threading import Thread


users = set()
bot = telebot.TeleBot("Your_TOKEN") #Замените Your_TOKEN на ваш токен бота, если не знаете как его получить - посмотрите видео о создании тг ботов
@bot.message_handler(commands=['start'])
def start(message):
    users.add(message.chat.id)
    bot.send_message(message.chat.id, 'Здравствуйте, каждый день в 18:00 вам будет приходить сообщение с погодой на следующий день')  

window_size = 2  # Размер окна, например 2 предыдущих дня
model = Sequential()
model.add(Dense(10, input_dim=window_size, activation="relu"))  # Теперь входной размер = размер окна
model.add(Dense(1))  # Выходной слой с одним значением
model.compile(loss='mse', optimizer='adam')

def learn():
    global model, window_size, all_we_x
    # Установим начальную и конечную даты
    start = datetime.datetime(2025, 1, 1) # Вместо 2025, 1, 1 введите дату, с которой будет начинаться отсчет температуры
    end = datetime.datetime.today()
    place = Point(55.755820, 37.617633) # Вместо 55.755820, 37.617633 введите координаты точки, для которой вы делаете прогноз, их можно получить на Яндекс картах
    data = Daily(place, start, end)
    data = data.fetch()

    # Используем только столбец с температурой
    data = data[['tavg']].reset_index()

    # Создадим временные окна для предсказания
    

        # Списки для входных и выходных данных
    pred_we_x = []
    next_we_y = []

        # Заполняем списки, используя данные с окнами
    for i in range(len(data) - window_size):
        pred_we_x.append(data['tavg'].iloc[i:i + window_size].values)  # Текущие 2 дня
        next_we_y.append(data['tavg'].iloc[i + window_size])  # Температура следующего дня
        pred_we_x_train = np.array(pred_we_x)
        next_we_y_train = np.array(next_we_y)
        model.fit(pred_we_x_train, next_we_y_train, epochs=100, batch_size=4096, verbose=0)
        print(f"round {i}")

    all_we_x = pred_we_x

        # Преобразуем в numpy массивы
    pred_we_x_train = np.array(pred_we_x)
    all_we_x.append(list(next_we_y[-2:]))
    next_we_y_train = np.array(next_we_y)
    all_we_x = np.array(all_we_x)

        # Обучение модели
    model.fit(pred_we_x_train, next_we_y_train, epochs=1000, batch_size=4096, verbose=0)
    print("learn was good")

    print("pred_we_x:", pred_we_x_train)
    print("next_we_y:", next_we_y_train)
    print("all_we_x:", all_we_x)

def send():
    global model, all_we_x
    predikt = model.predict(all_we_x[-1].reshape(1, -1))
    for user in list(users):
        bot.send_message(user, f"Температура завтра: {int(predikt)}")


def scheduler():
    schedule.every().day.at("12:00").do(learn)  # Обучение нейросети в 12:00
    schedule.every().day.at("18:00").do(send)  # Отправка прогноза в 18:00

    while True:
        schedule.run_pending()  # Проверяем запланированные задачи
        time.sleep(1)  # Спим 1 секунду, чтобы не нагружать процессор

scheduler_thread = Thread(target=scheduler)
scheduler_thread.daemon = True  # Поток завершится при завершении основного процесса
scheduler_thread.start()


bot.polling(none_stop=True)
