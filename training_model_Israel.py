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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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
analyzer = SentimentIntensityAnalyzer()
tweets["sentiment"] = tweets["text"].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# 4. ממוצע סנטימנט לכל מתמודד
def match_contestant(text, contestant_names):
    for name in contestant_names:
        if pd.isna(name):
            continue
        if str(name).lower() in str(text).lower():
            return name
    return None

tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, contestants["full_name"]))
tweets_filtered = tweets.dropna(subset=["username"])

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

data.loc[X.index, "predicted_score"] = all_data_preds
ranking = data.loc[X.index].sort_values(by="predicted_score", ascending=False)

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

