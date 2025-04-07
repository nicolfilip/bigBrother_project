import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# טען את הנתונים
contestants = pd.read_csv("big_brother_israel.csv")

print(contestants.columns.tolist())

tweets = pd.read_csv("big_brother_tweets_ISRAEL.csv")

# נוודא שעמודת 'שם מלא' קיימת ונכונה
contestants["full_name"] = contestants["שם מלא"].astype(str)

# ניתוח סנטימנט עם VADER
analyzer = SentimentIntensityAnalyzer()
tweets["sentiment"] = tweets["text"].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# התאמת ציוצים למתמודדים
def match_contestant(text, contestant_names):
    for name in contestant_names:
        if pd.isna(name):
            continue
        if str(name).lower() in str(text).lower():
            return name
    return None

tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, contestants["full_name"]))
tweets_filtered = tweets.dropna(subset=["username"])

# ממוצע סנטימנט לכל מתמודד
avg_sentiment = tweets_filtered.groupby("username")["sentiment"].mean().reset_index()
avg_sentiment.columns = ["full_name", "avg_sentiment"]

# מיזוג עם טבלת המתמודדים
data = contestants.merge(avg_sentiment, on="full_name", how="left")
data["avg_sentiment"] = data["avg_sentiment"].fillna(0)

# ניצור עמודת דירוג סינתטי לפי הסדר בכל עונה
data["final_placement"] = data.groupby("1").cumcount() + 1

# חישוב ניקוד דירוג – ככל שהדירוג נמוך יותר (הודח מוקדם), הציון נמוך יותר
data["rank_score"] = data["final_placement"].max() - data["final_placement"] + 1

data["גיל"] = pd.to_numeric(data["גיל"], errors="coerce")
X = data[["גיל", "avg_sentiment"]].dropna()
y = data.loc[X.index, "rank_score"]

# חלוקה לרכבת/בדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# המרת הנתונים ל-DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# הגדרת פרמטרים
params = {
    "objective": "rank:pairwise",
    "learning_rate": 0.1,
    "max_depth": 4
}

# אימון המודל
model = xgb.train(params, dtrain, num_boost_round=100)

# תחזית ו-nDCG
y_pred = model.predict(dtest)
print("nDCG:", ndcg_score([y_test], [y_pred]))

# דירוג כל הנתונים
all_data_dmatrix = xgb.DMatrix(X)
all_data_preds = model.predict(all_data_dmatrix)

data.loc[X.index, "predicted_score"] = all_data_preds
ranking = data.loc[X.index].sort_values(by="predicted_score", ascending=False)

# הדפסת Top 10
top_10 = ranking[["full_name", "final_placement", "avg_sentiment", "predicted_score"]].head(10)
print(top_10)
