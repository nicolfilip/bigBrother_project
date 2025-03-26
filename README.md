# 🎯 Big Brother Elimination Prediction

A machine learning-based system that predicts the next contestant to be eliminated from the reality TV show **Big Brother**.  
The system analyzes both **static contestant data** and **dynamic public sentiment from Twitter** to train a predictive model.

---

## 🧠 Key Features

- Analyzes contestant attributes (age, gender, competition stats).
- Connects to the Twitter API to fetch live tweets mentioning contestants.
- Performs sentiment analysis on tweets using TextBlob.
- Calculates average sentiment scores per contestant.
- Trains a basic classification model (Random Forest) using features like age and sentiment.
- Supports three scenarios: USA contestants, Israeli contestants, and combined data.
- Modular and ready for upgrades with LSTM, TF-IDF, or more advanced NLP.

---

