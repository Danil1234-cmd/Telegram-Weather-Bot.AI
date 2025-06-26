# bot.py
import telebot
import schedule
import time
from threading import Thread
from min_temp_model import init_model as min_init, learn_min, predict_min
from avg_temp_model import init_model as avg_init, learn_avg, predict_avg
from max_temp_model import init_model as max_init, learn_max, predict_max

users = set()
bot = telebot.TeleBot("YOUR_API_TOKEN")

@bot.message_handler(commands=['start'])
def start(message):
    users.add(message.chat.id)
    bot.send_message(message.chat.id, 'Hello! Every day at 18:00 you will receive the weather forecast.')
    


def learn_all():
    learn_min()
    learn_avg()
    learn_max()
    print("All models have been successfully trained.")



def send_forecast():
    min_temp = predict_min()
    avg_temp = predict_avg()
    max_temp = predict_max()
    
    if None in (min_temp, avg_temp, max_temp):
        return
    
    message = (
        "Weather forecast for tomorrow:\n"
        f"• Min: {min_temp:.1f}°C\n"
        f"• Avg: {avg_temp:.1f}°C\n"
        f"• Max: {max_temp:.1f}°C"
    )
    
    for user in users:
        bot.send_message(user, message)

def scheduler():
    min_init()
    avg_init()
    max_init()
    
    schedule.every().day.at("12:00").do(learn_all)
    schedule.every().day.at("18:00").do(send_forecast)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
    

if __name__ == "__main__":
    scheduler_thread = Thread(target=scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    bot.polling(none_stop=True)