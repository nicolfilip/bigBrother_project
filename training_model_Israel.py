# import pandas as pd
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import ndcg_score
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from itertools import combinations
# import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
#
#
# # טען את הנתונים
# contestants = pd.read_csv("big_brother_israel.csv")
#
# print(contestants.columns.tolist())
#
# tweets = pd.read_csv("big_brother_tweets_ISRAEL.csv")
#
# # נוודא שעמודת 'שם מלא' קיימת ונכונה
# contestants["full_name"] = contestants["שם מלא"].astype(str)
#
# # ניתוח סנטימנט עם VADER
# analyzer = SentimentIntensityAnalyzer()
# tweets["sentiment"] = tweets["text"].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
#
# # התאמת ציוצים למתמודדים
# def match_contestant(text, contestant_names):
#     for name in contestant_names:
#         if pd.isna(name):
#             continue
#         if str(name).lower() in str(text).lower():
#             return name
#     return None
#
# tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, contestants["full_name"]))
# tweets_filtered = tweets.dropna(subset=["username"])
#
# # ממוצע סנטימנט לכל מתמודד
# avg_sentiment = tweets_filtered.groupby("username")["sentiment"].mean().reset_index()
# avg_sentiment.columns = ["full_name", "avg_sentiment"]
#
# # מיזוג עם טבלת המתמודדים
# data = contestants.merge(avg_sentiment, on="full_name", how="left")
# data["avg_sentiment"] = data["avg_sentiment"].fillna(0)
#
# # ניצור עמודת דירוג סינתטי לפי הסדר בכל עונה
# data["final_placement"] = data.groupby("1").cumcount() + 1
#
# # חישוב ניקוד דירוג – ככל שהדירוג נמוך יותר (הודח מוקדם), הציון נמוך יותר
# data["rank_score"] = data["final_placement"].max() - data["final_placement"] + 1
#
# data["גיל"] = pd.to_numeric(data["גיל"], errors="coerce")
# X = data[["גיל", "avg_sentiment"]].dropna()
# y = data.loc[X.index, "rank_score"]
#
# # חלוקה לרכבת/בדיקה
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # המרת הנתונים ל-DMatrix
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)
#
# # הגדרת פרמטרים
# params = {
#     "objective": "rank:pairwise",
#     "learning_rate": 0.1,
#     "max_depth": 4
# }
#
# # אימון המודל
# model = xgb.train(params, dtrain, num_boost_round=100)
#
# # תחזית ו-nDCG
# y_pred = model.predict(dtest)
# print("nDCG:", ndcg_score([y_test], [y_pred]))
#
# # דירוג כל הנתונים
# all_data_dmatrix = xgb.DMatrix(X)
# all_data_preds = model.predict(all_data_dmatrix)
#
# data.loc[X.index, "predicted_score"] = all_data_preds
# ranking = data.loc[X.index].sort_values(by="predicted_score", ascending=False)
#
# # הדפסת Top 10
# top_10 = ranking[["full_name", "final_placement", "avg_sentiment", "predicted_score"]].head(10)
# print(top_10)
#
#
#
# # יצירת רשימת קשתות
# edges = []
#
# # נעבור על כל הציוצים
# for _, tweet in tweets.iterrows():
#     # נזהה איזה מתמודדים מופיעים בציוץ
#     mentioned = [name for name in contestants["full_name"] if pd.notna(name) and str(name) in tweet["text"]]
#     if len(mentioned) >= 2:
#         for pair in combinations(mentioned, 2):
#             edges.append({
#                 "from": pair[0],
#                 "to": pair[1],
#                 "weight": abs(tweet["sentiment"]),
#                 "type": "positive" if tweet["sentiment"] > 0 else "negative"
#             })
#
# # המרת קשתות לדאטהפריים
# edges_df = pd.DataFrame(edges)
#
# # שמירה לקובץ (אם תרצי)
# edges_df.to_csv("israel_graph_edges.csv", index=False, encoding="utf-8")
#
# # יצירת גרף
# G = nx.Graph()
#
# # הוספת קשתות לגרף
# for _, row in edges_df.iterrows():
#     G.add_edge(row["from"], row["to"], weight=row["weight"], sentiment=row["type"])
#
# # ציור הגרף
# pos = nx.spring_layout(G, seed=42)  # פריסת גרף אסתטית
# edge_colors = ['green' if G[u][v]['sentiment'] == 'positive' else 'red' for u, v in G.edges()]
#
# plt.figure(figsize=(14, 10))
# nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue', node_size=1500, font_size=10, width=2)
# plt.title("גרף קשרים בין מתמודדי האח הגדול")
# plt.show()
#
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
<<<<<<< HEAD
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
=======
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import os

# 1. קריאת קבצים
contestants = pd.read_csv("big_brother_israel.csv")
tweets = pd.read_csv("big_brother_tweets_ISRAEL.csv")

# 2. הכנת עמודת שמות
contestants["full_name"] = contestants["שם מלא"].astype(str)

# 3. ניתוח סנטימנט
>>>>>>> 3d3b309d6ef8b154136d206b0fbe45f05e866f51
analyzer = SentimentIntensityAnalyzer()
tweets["sentiment"] = tweets["text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])

<<<<<<< HEAD
=======
# 4. ממוצע סנטימנט לכל מתמודד
>>>>>>> 3d3b309d6ef8b154136d206b0fbe45f05e866f51
def match_contestant(text, contestant_names):
    for name in contestant_names:
        if pd.isna(name):
            continue
        if str(name).lower() in str(text).lower():
            return name
    return None

tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, df["full_name"]))
tweets_filtered = tweets.dropna(subset=["username"])

<<<<<<< HEAD
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
=======
avg_sentiment = tweets_filtered.groupby("username")["sentiment"].mean().reset_index()
avg_sentiment.columns = ["full_name", "avg_sentiment"]

# 5. מיזוג נתונים
data = contestants.merge(avg_sentiment, on="full_name", how="left")
data["avg_sentiment"] = data["avg_sentiment"].fillna(0)

data["final_placement"] = data.groupby("1").cumcount() + 1
data["rank_score"] = data["final_placement"].max() - data["final_placement"] + 1
data["גיל"] = pd.to_numeric(data["גיל"], errors="coerce")

X = data[["גיל", "avg_sentiment"]].dropna()
y = data.loc[X.index, "rank_score"]

# 6. אימון מודל
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "rank:pairwise",
    "learning_rate": 0.1,
    "max_depth": 4
}

model = xgb.train(params, dtrain, num_boost_round=100)

y_pred = model.predict(dtest)
print("nDCG:", ndcg_score([y_test], [y_pred]))

all_data_dmatrix = xgb.DMatrix(X)
all_data_preds = model.predict(all_data_dmatrix)
>>>>>>> 3d3b309d6ef8b154136d206b0fbe45f05e866f51


<<<<<<< HEAD
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
=======
top_10 = ranking[["full_name", "final_placement", "avg_sentiment", "predicted_score"]].head(10)
print(top_10)

# ------------------------
# 7. גרף קשרים ידני מקובץ CSV
# ------------------------

# טען את קובץ הקשרים הידני שיצרת
manual_edges = pd.read_csv("graph_output/Untitled spreadsheet - Sheet1.csv")

# יצירת הגרף
G = nx.Graph()

# הוספת הקשרים לגרף
for _, row in manual_edges.iterrows():
    G.add_edge(row["from"], row["to"], weight=row["weight"], sentiment=row["sentiment"])

# הגדרות צבעים לפי סנטימנט
edge_colors = ["green" if G[u][v]["sentiment"] == "positive" else "red" for u, v in G.edges()]

# ציור
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(16, 10))
nx.draw(
    G, pos,
    with_labels=True,
    edge_color=edge_colors,
    node_color="lightblue",
    node_size=1500,
    font_size=10,
    width=2
)
plt.title("גרף קשרים ידני – עונה 12")
plt.savefig("graph_output/bigbrother_manual_graph.png", dpi=300)
plt.show()

print("גרף שמור כקובץ: graph_output/bigbrother_manual_graph.png")

>>>>>>> 3d3b309d6ef8b154136d206b0fbe45f05e866f51
