import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

# קריאת קבצים
contestants = pd.read_csv("big_brother_israel_cleaned.csv", header=0)
contestants = contestants.loc[:, ~contestants.columns.str.contains('^Unnamed')]
tweets = pd.read_csv("../data/big_brother_tweets_ISRAEL.csv")

# עיבוד שמות
contestants["full_name"] = contestants["שם מלא"].astype(str)

# ניתוח סנטימנט
analyzer = SentimentIntensityAnalyzer()
tweets["sentiment"] = tweets["text"].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])


# התאמת ציוצים לשמות
def match_contestant(text, contestant_names):
    for name in contestant_names:
        if pd.isna(name):
            continue
        if str(name).lower() in str(text).lower():
            return name
    return None


tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, contestants["full_name"]))
tweets_filtered = tweets.dropna(subset=["username"])

# תכונות ממוצעות
avg_sentiment = tweets_filtered.groupby("username")["sentiment"].mean().reset_index()
avg_sentiment.columns = ["full_name", "avg_sentiment"]
mention_counts = tweets_filtered["username"].value_counts().reset_index()
mention_counts.columns = ["full_name", "mention_count"]
sentiment_std = tweets_filtered.groupby("username")["sentiment"].std().reset_index()
sentiment_std.columns = ["full_name", "sentiment_std"]

# מיזוג מידע
data = contestants.merge(avg_sentiment, on="full_name", how="left")
data = data.merge(mention_counts, on="full_name", how="left")
data = data.merge(sentiment_std, on="full_name", how="left")

# מילוי ערכים חסרים
data["avg_sentiment"] = data["avg_sentiment"].fillna(0)
data["mention_count"] = data["mention_count"].fillna(0)
data["sentiment_std"] = data["sentiment_std"].fillna(0)

# קידוד תכונות
data["gender_encoded"] = data["מין"].map({"ז": 1, "נ": 0})
data["status_encoded"] = data["סטטוס"].astype(str).apply(lambda s: 1 if "נשוי" in s else 0)
data["is_vip"] = data["שם מלא"].astype(str).apply(lambda x: 1 if "VIP" in x or "מהעונה" in x else 0)
data = data.dropna(subset=["Days in game"])
data["Days in game"] = pd.to_numeric(data["Days in game"], errors="coerce")


data["rank"] = data["Days in game"].rank(ascending=False, method="min")
data["rank"] = data["rank"].astype(int)

max_days = data["Days in game"].max()


def classify_by_days(days):
    if pd.isna(days):
        return None
    if days <= 0.25 * max_days:
        return 0
    elif days <= 0.5 * max_days:
        return 1
    elif days <= 0.75 * max_days:
        return 2
    else:
        return 3


data["stage_class"] = data["Days in game"].apply(classify_by_days)
data = data.drop(columns=["Days in game", "rank"])

# בחירת פיצ'רים
X = data[["גיל", "avg_sentiment", "mention_count", "sentiment_std",
          "gender_encoded", "status_encoded", "is_vip"]].dropna()

y = data.loc[X.index, "stage_class"]
X = X[y.notna()]
y = y[y.notna()]

#split the train and test
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#check
overlap = set(X_train.index) & set(X_test.index)
print(f"\n🔍 Overlap in indices: {len(overlap)}")

#balance the categories in the train
oversampler = RandomOverSampler(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

correlations = data.corr(numeric_only=True)
print(correlations["stage_class"].sort_values(ascending=False))

#train the model
model = XGBClassifier(objective="multi:softmax", num_class=4, eval_metric="mlogloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# תוצאות
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# שמירת תוצאות
output_path = "big_brother_israel_with_predictions.csv"
data.loc[X.index, "predicted_stage"] = model.predict(X)
data.to_csv(output_path, index=False, encoding="utf-8")
print(f"\nSaved predictions to {output_path}")
