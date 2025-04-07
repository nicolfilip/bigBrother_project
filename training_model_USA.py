# Assignment 1:
# Nicol Filipchuk 206637985
# Yuval Malka 315402669

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Load data
contestants = pd.read_csv("big_brother_usa.csv")
tweets = pd.read_csv("big_brother_tweets_USA.csv")

# Create full name
contestants["full_name"] = contestants["first"].astype(str) + " " + contestants["last"].astype(str)

# Analyze sentiment with VADER
analyzer = SentimentIntensityAnalyzer()
tweets["sentiment"] = tweets["text"].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Match tweets to contestants
def match_contestant(text, contestant_names):
    for name in contestant_names:
        if pd.isna(name):
            continue
        if str(name).lower() in str(text).lower():
            return name
    return None

tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, contestants["full_name"]))
tweets_filtered = tweets.dropna(subset=["username"])

# Compute average sentiment per contestant
avg_sentiment = tweets_filtered.groupby("username")["sentiment"].mean().reset_index()
avg_sentiment.columns = ["full_name", "avg_sentiment"]

# Merge with contestant data
data = contestants.merge(avg_sentiment, on="full_name", how="left")
data["avg_sentiment"] = data["avg_sentiment"].fillna(0)

# Rank score = higher placement = higher score
data["rank_score"] = data["final_placement"].max() - data["final_placement"] + 1

# Features and labels
X = data[["age", "avg_sentiment"]].dropna()
y = data.loc[X.index, "rank_score"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters (removed n_estimators – use num_boost_round instead)
params = {
    "objective": "rank:pairwise",
    "learning_rate": 0.1,
    "max_depth": 4
}

# Train model
model = xgb.train(params, dtrain, num_boost_round=100)

# Predict
y_pred = model.predict(dtest)
print("nDCG:", ndcg_score([y_test], [y_pred]))

# Predict full data
all_data_dmatrix = xgb.DMatrix(X)
all_data_preds = model.predict(all_data_dmatrix)

data.loc[X.index, "predicted_score"] = all_data_preds
ranking = data.loc[X.index].sort_values(by="predicted_score", ascending=False)

# Print Top 10
top_10 = ranking[["full_name", "final_placement", "avg_sentiment", "predicted_score"]].head(10)
print(top_10)
