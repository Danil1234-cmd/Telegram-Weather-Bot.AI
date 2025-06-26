# Weather Forecast Telegram Bot 🌦️

A Telegram bot that predicts next-day weather using neural networks. The bot forecasts min, avg, and max temperatures for Moscow with deep learning models trained on historical weather data.

## Features ✨
- **Neural Network Prediction**: Uses TensorFlow/Keras models trained on real weather data
- **Daily Forecasts**: Automatic updates sent daily at 18:00 local time
- **Moscow Coverage**: Focused on coordinates 55.755820, 37.617633 (Moscow)
- **Three Separate Models**:
  - Minimum temperature prediction (`tmin`)
  - Average temperature prediction (`tavg`)
  - Maximum temperature prediction (`tmax`)
- **Automatic Training**: Models retrain daily at 12:00 using fresh data

## Technical Implementation 🔧
- **Window Size**: 2-day sliding window for predictions
- **Model Architecture**:
  ```python
  Sequential([
    Dense(10, input_dim=window_size, activation='relu'),
    Dense(1)
  ])

## Installation & Setup 🚀
### Clone repository:

```bash
git clone https://github.com/Danil1234-cmd/weather-forecast-bot.git
cd weather-forecast-bot
```
### Create virtual environment:


```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

### Create a bot via @BotFather

Replace YOUR_API_TOKEN in bot.py with your token

### Run the bot:

```bash
./start.sh  # Linux/Mac
start.bat   # Windows alternative
