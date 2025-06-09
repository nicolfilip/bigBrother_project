# ייבוא ספריות
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker
import networkx as nx
from transformers import pipeline

# --- טעינת הדאטה --- #;
print("1")

# נתוני משתתפים
df = pd.read_csv("big_brother_israel_new_new.csv")
df["full_name"] = df["שם מלא"].astype(str)

# נתוני ציוצים
tweets = pd.read_csv("big_brother_tweets_ISRAEL.csv")

# נתוני גרף קשרים
edges_df = pd.read_csv("graph_output/Untitled spreadsheet - Sheet1.csv")

# --- בניית גרף חברתי --- #

G = nx.DiGraph()
for _, row in edges_df.iterrows():
    G.add_edge(row["from"], row["to"], weight=row["weight"], sentiment=row["sentiment"])

# חישוב מאפייני גרף
pagerank = nx.pagerank(G)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
in_degree = dict(G.in_degree(weight="weight"))
out_degree = dict(G.out_degree(weight="weight"))
total_degree = dict(G.degree(weight="weight"))

sentiment_weights = {s: {} for s in ["positive", "negative", "romantic", "netural"]}
for u, v, d in G.edges(data=True):
    sentiment = d.get("sentiment", "netural")
    weight = d.get("weight", 1)
    if sentiment in sentiment_weights:
        sentiment_weights[sentiment][u] = sentiment_weights[sentiment].get(u, 0) + weight

# בניית DataFrame חברתי
social_df = pd.DataFrame({"full_name": df["full_name"]}).assign(
    pagerank=lambda x: x["full_name"].map(pagerank).fillna(0),
    betweenness=lambda x: x["full_name"].map(betweenness).fillna(0),
    closeness=lambda x: x["full_name"].map(closeness).fillna(0),
    in_degree=lambda x: x["full_name"].map(in_degree).fillna(0),
    out_degree=lambda x: x["full_name"].map(out_degree).fillna(0),
    total_degree=lambda x: x["full_name"].map(total_degree).fillna(0),
    **{f"{s}_out_weight": lambda x, s=s: x["full_name"].map(sentiment_weights[s]).fillna(0) for s in sentiment_weights}
)

df = df.merge(social_df, on="full_name", how="left")

# --- ניקוי ויצירת פיצ'רים --- #

# הפיכת עמודות למספריות
df["Days in game"] = pd.to_numeric(df["Days in game"], errors="coerce")
df["week eliminated"] = pd.to_numeric(df["week eliminated"], errors="coerce")
df["entered week"] = pd.to_numeric(df["entered week"], errors="coerce")
df["rank_score"] = df["week eliminated"] - df["entered week"]

# קידוד מגדר, סטטוס, VIP
df["gender_encoded"] = df["מין"].map({"ז": 1, "נ": 0})
df["status_encoded"] = df["סטטוס"].astype(str).apply(lambda s: 1 if "נשוי" in s else 0)
df["is_vip"] = df["שם מלא"].astype(str).apply(lambda x: 1 if "VIP" in x or "מהעונה" in x else 0)

# קידוד עונה
df["season_id"] = LabelEncoder().fit_transform(df["עונה"])

# --- ניתוח סנטימנט בעברית עם HeBERT --- #

print("🔍 Loading HeBERT sentiment model... (one-time download)")
sentiment_pipeline = pipeline("sentiment-analysis", model="avichr/heBERT_sentiment_analysis")

# פונקציה להערכת סנטימנט על ציוץ
def get_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])  # חיתוך ל-512 תווים
        label = result[0]['label']
        # נתרגם ל-POSITIVE/NEGATIVE מספרי
        return 1 if label == "Positive" else -1
    except:
        return 0  # במקרה של שגיאה

tweets["sentiment"] = tweets["text"].apply(get_sentiment)

# --- שיוך ציוצים למתמודדים --- #

def match_contestant(text, contestant_names):
    for name in contestant_names:
        if pd.isna(name):
            continue
        if str(name).lower() in str(text).lower():
            return name
    return None

tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, df["full_name"]))
tweets_filtered = tweets.dropna(subset=["username"])

# --- בניית פיצ'רים מהציוצים --- #

if not tweets_filtered.empty:
    avg_sentiment = tweets_filtered.groupby("username")["sentiment"].mean().reset_index(name="avg_sentiment")
    mention_counts = tweets_filtered["username"].value_counts().reset_index()
    mention_counts.columns = ["full_name", "mention_count"]
    sentiment_std = tweets_filtered.groupby("username")["sentiment"].std().reset_index(name="sentiment_std")

    df = df.merge(avg_sentiment, left_on="full_name", right_on="username", how="left").drop(columns="username")
    df = df.merge(mention_counts, on="full_name", how="left")
    df = df.merge(sentiment_std, left_on="full_name", right_on="username", how="left").drop(columns="username")
else:
    df["avg_sentiment"] = 0
    df["mention_count"] = 0
    df["sentiment_std"] = 0

df[["avg_sentiment", "mention_count", "sentiment_std"]] = df[["avg_sentiment", "mention_count", "sentiment_std"]].fillna(0)

# עמודת גיל
df["גיל"] = pd.to_numeric(df["גיל"], errors="coerce")

# רעש קל (noise)
np.random.seed(42)
noise_strength = {"avg_sentiment": 0.05, "mention_count": 2.0, "sentiment_std": 0.05, "גיל": 0.5}
for col, strength in noise_strength.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] += np.random.normal(0, strength, size=len(df))

# --- בניית מודל דירוג --- #

features = [
    "גיל", "gender_encoded", "status_encoded", "is_vip",
    "avg_sentiment", "mention_count", "sentiment_std",
    "pagerank", "betweenness", "closeness",
    "in_degree", "out_degree", "total_degree",
    "positive_out_weight", "negative_out_weight",
    "romantic_out_weight", "netural_out_weight"
]

df_train = df[df["rank_score"].notna()].copy()
df_predict = df[df["rank_score"].isna()].copy()

X = df_train[features]
y = df_train["rank_score"]
groups = df_train["season_id"]

cv = GroupKFold(n_splits=5)
scores = []
preds_all, true_all = [], []

# קרוס ולידציה
for train_idx, test_idx in cv.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    group_train = df_train.iloc[train_idx].groupby("season_id").size().to_list()

    model = XGBRanker(objective="rank:pairwise", n_estimators=100, random_state=42)
    model.fit(X_train, y_train, group=group_train)

    preds = model.predict(X_test)
    preds_all.extend(preds)
    true_all.extend(y_test.values)
    scores.append(ndcg_score([y_test.values], [preds]))

print("📊 Average nDCG across folds:", np.mean(scores))

# --- חיזוי על עונה 15 --- #

season_15_df = df_predict[df_predict["עונה"] == "15"].copy()
if not season_15_df.empty:
    X_alive = season_15_df[features]
    preds_alive = model.predict(X_alive)
    season_15_df["predicted_score"] = preds_alive
    eliminated_next = season_15_df.loc[season_15_df["predicted_score"].idxmin()]
    print(f"\n🔮 Predicted next to be eliminated in Season 15: {eliminated_next['full_name']}")
    print("\n📋 Top 3 candidates at risk:")
    print(season_15_df.sort_values("predicted_score")[['full_name', 'predicted_score']].head(3))
else:
    print("⚠️ No active candidates found for season 15")

# --- גרף תוצאה --- #

plt.figure(figsize=(10, 6))
plt.scatter(true_all, preds_all, alpha=0.6)
plt.xlabel("True Rank Score")
plt.ylabel("Predicted")
plt.title("Predicted vs. True Rank Scores")
plt.grid(True)
plt.tight_layout()
plt.show()
