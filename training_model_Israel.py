import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scores
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xgboost import XGBRanker

# Load datasets
df = pd.read_csv("big_brother_israel_cleaned.csv")
tweets = pd.read_csv("big_brother_tweets_ISRAEL.csv")

#delete the rows that don't have information about one of those
df = df.dropna(subset=["Days in game", "עונה"])
df["Days in game"] = pd.to_numeric(df["Days in game"], errors="coerce")
df = df.dropna(subset=["Days in game"])
df["full_name"] = df["שם מלא"].astype(str)

# Feature Engineering
df["gender_encoded"] = df["מין"].map({"ז": 1, "נ": 0})
df["status_encoded"] = df["סטטוס"].astype(str).apply(lambda s: 1 if "נשוי" in s else 0)
df["is_vip"] = df["שם מלא"].astype(str).apply(lambda x: 1 if "VIP" in x or "מהעונה" in x else 0)
df["season_id"] = LabelEncoder().fit_transform(df["עונה"])
df["rank_score"] = df["Days in game"]

# Sentiment from tweets
analyzer = SentimentIntensityAnalyzer()
tweets["sentiment"] = tweets["text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])

def match_contestant(text, contestant_names):
    for name in contestant_names:
        if pd.isna(name):
            continue
        if str(name).lower() in str(text).lower():
            return name
    return None

tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, df["full_name"]))
tweets_filtered = tweets.dropna(subset=["username"])

# Aggregate tweet features
avg_sentiment = tweets_filtered.groupby("username")["sentiment"].mean().reset_index()
avg_sentiment.columns = ["full_name", "avg_sentiment"]

mention_counts = tweets_filtered["username"].value_counts().reset_index()
mention_counts.columns = ["full_name", "mention_count"]

sentiment_std = tweets_filtered.groupby("username")["sentiment"].std().reset_index()
sentiment_std.columns = ["full_name", "sentiment_std"]

# Merge with main dataframe
df = df.merge(avg_sentiment, on="full_name", how="left")
df = df.merge(mention_counts, on="full_name", how="left")
df = df.merge(sentiment_std, on="full_name", how="left")
df[["avg_sentiment", "mention_count", "sentiment_std"]] = df[
    ["avg_sentiment", "mention_count", "sentiment_std"]
].fillna(0)

#add noise to avoid of overFitting
#Restoring results
np.random.seed(42)

noise_strength = {
    "avg_sentiment": 0.05,
    "mention_count": 2.0,
    "sentiment_std": 0.05,
    "גיל": 0.5
}

for col, strength in noise_strength.items():
    if col in df.columns:
        df[col] = df[col] + np.random.normal(0, strength, size=len(df))


# Final feature matrix
features = [
    "גיל", "gender_encoded", "status_encoded", "is_vip",
    "avg_sentiment", "mention_count", "sentiment_std"
]
X = df[features]
y = df["rank_score"]
groups = df["season_id"]


gkf = GroupKFold(n_splits=5)
scores = []
all_preds = []
all_true = []

for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    group_train = df.iloc[train_idx].groupby("season_id").size().to_list()

    ranker = XGBRanker(objective="rank:pairwise", random_state=42, n_estimators=25)
    ranker.fit(X_train, y_train, group=group_train)

    preds = ranker.predict(X_test)
    scores.append(ndcg_score([y_test.values], [preds]))

    all_preds.extend(preds)
    all_true.extend(y_test.values)

# Plot predicted vs. true
plt.figure(figsize=(10, 6))
plt.scatter(all_true, all_preds, alpha=0.6)
plt.xlabel("True Rank Score (Days in Game)")
plt.ylabel("Predicted Rank Score")
plt.title("Predicted vs. True Rank Scores (XGBRanker with Noise + CV)")
plt.grid(True)
plt.tight_layout()

# Return average nDCG
np.mean(scores)
# Return average nDCG
print("Average nDCG across folds:", np.mean(scores))

# Show plot
plt.show()
