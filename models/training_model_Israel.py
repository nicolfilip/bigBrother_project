import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker
from lightgbm import LGBMRanker
import networkx as nx
from transformers import pipeline
from PIL import Image
import os


def show_contestant_image(name):
    extensions = ['jpg', 'jpeg', 'png', 'webp']
    found = False

    for ext in extensions:
        image_path = f"images/{name}.{ext}"
        if os.path.isfile(image_path):
            found = True
            break

    if found:
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print(f"לא נמצאה תמונה עבור {name}")




df = pd.read_csv("../data/big_brother_israel_new_new.csv")
df["full_name"] = df["שם מלא"].astype(str)

tweets = pd.read_csv("../data/big_brother_tweets_ISRAEL.csv")

edges_df = pd.read_csv("../graph_output/season15/graph_heb.csv")

G = nx.DiGraph()
for _, row in edges_df.iterrows():
    G.add_edge(row["from"], row["to"], weight=row["weight"], sentiment=row["sentiment"])

pagerank = nx.pagerank(G)
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
in_degree = dict(G.in_degree(weight="weight"))
out_degree = dict(G.out_degree(weight="weight"))
total_degree = dict(G.degree(weight="weight"))

sentiment_weights = {s: {} for s in ["positive", "negative", "romantic", "neutral"]}
for u, v, d in G.edges(data=True):
    sentiment = d.get("sentiment", "neutral")
    weight = d.get("weight", 1)
    if sentiment in sentiment_weights:
        sentiment_weights[sentiment][u] = sentiment_weights[sentiment].get(u, 0) + weight

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

df["Days in game"] = pd.to_numeric(df["Days in game"], errors="coerce")
df["week eliminated"] = pd.to_numeric(df["week eliminated"], errors="coerce")
df["entered week"] = pd.to_numeric(df["entered week"], errors="coerce")
df["rank_score"] = df["week eliminated"] - df["entered week"]

df["gender_encoded"] = df["מין"].map({"ז": 1, "נ": 0})
df["status_encoded"] = df["סטטוס"].astype(str).apply(lambda s: 1 if "נשוי" in s else 0)
df["is_vip"] = df["שם מלא"].astype(str).apply(lambda x: 1 if "VIP" in x or "מהעונה" in x else 0)

df["season_id"] = LabelEncoder().fit_transform(df["עונה"])

print("Loading HeBERT sentiment model...")
sentiment_pipeline = pipeline("sentiment-analysis", model="avichr/heBERT_sentiment_analysis")


def get_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])
        label = result[0]['label']
        return 1 if label == "Positive" else -1
    except:
        return 0


tweets["sentiment"] = tweets["text"].apply(get_sentiment)


def match_contestant(text, contestant_names):
    for name in contestant_names:
        if pd.isna(name):
            continue
        if str(name).lower() in str(text).lower():
            return name
    return None


tweets["username"] = tweets["text"].apply(lambda x: match_contestant(x, df["full_name"]))
tweets_filtered = tweets.dropna(subset=["username"])

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

df[["avg_sentiment", "mention_count", "sentiment_std"]] = df[
    ["avg_sentiment", "mention_count", "sentiment_std"]].fillna(0)

df["גיל"] = pd.to_numeric(df["גיל"], errors="coerce")

np.random.seed(42)
noise_strength = {"avg_sentiment": 0.05, "mention_count": 2.0, "sentiment_std": 0.05, "גיל": 0.5}
for col, strength in noise_strength.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] += np.random.normal(0, strength, size=len(df))

features = [
    "גיל", "gender_encoded", "status_encoded", "is_vip",
    "avg_sentiment", "mention_count", "sentiment_std",
    "pagerank", "betweenness", "closeness",
    "in_degree", "out_degree", "total_degree",
    "positive_out_weight", "negative_out_weight",
    "romantic_out_weight", "neutral_out_weight"
]

df_train = df[df["rank_score"].notna()].copy()
df_predict = df[df["rank_score"].isna()].copy()

X = df_train[features]
y = df_train["rank_score"]
groups = df_train["season_id"]

cv = GroupKFold(n_splits=5)
scores = []
preds_all, true_all = [], []

for train_idx, test_idx in cv.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    group_train = df_train.iloc[train_idx].groupby("season_id").size().to_list()

    xgb_model = XGBRanker(objective="rank:pairwise", n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train, group=group_train)

    preds = xgb_model.predict(X_test)
    preds_all.extend(preds)
    true_all.extend(y_test.values)
    scores.append(ndcg_score([y_test.values], [preds]))

print("Average nDCG across folds (XGB):", np.mean(scores))

X_train_stacked = X.copy()
X_train_stacked["xgb_predicted"] = xgb_model.predict(X)

lgbm_model = LGBMRanker(objective="lambdarank", n_estimators=100, random_state=42, verbose=-1)
groups_lgb = df_train.groupby("season_id").size().to_list()
lgbm_model.fit(X_train_stacked, y, group=groups_lgb)

def run_model_on_nominees(nominees, return_top=False):
    season_15_df = df_predict[df_predict["עונה"] == "15"].copy()
    season_15_df = season_15_df[season_15_df["full_name"].isin(nominees)]

    if not season_15_df.empty:
        X_alive = season_15_df[features]
        X_alive = X_alive.copy()
        X_alive["xgb_predicted"] = xgb_model.predict(X_alive)

        preds_alive = lgbm_model.predict(X_alive)
        season_15_df["predicted_score"] = preds_alive

        eliminated_next = season_15_df.loc[season_15_df["predicted_score"].idxmin()]
        print(f"\nPredicted next to be eliminated in Season 15: {eliminated_next['full_name']}")

        # הצגת תמונה של המודח
        show_contestant_image(eliminated_next['full_name'])

        print("\nTop 3 candidates at risk:")
        print(season_15_df.sort_values("predicted_score")[['full_name', 'predicted_score']].head(3))
        if return_top:
            top_risk = season_15_df.sort_values("predicted_score")[['full_name', 'predicted_score']].head(3)
            top_list = list(zip(top_risk["full_name"], top_risk["predicted_score"]))
            return {"eliminated": eliminated_next["full_name"], "top_risk": top_list}
        else:
            return eliminated_next["full_name"]
    else:
        print("No nominees matched the contestants.")
        return None


if __name__ == "__main__":
    print("\nEnter the 6 nominees for eviction:")
    nominees = []
    while len(nominees) < 6:
        name = input(f"Nominee {len(nominees) + 1}: ").strip()
        if name:
            nominees.append(name)

    run_model_on_nominees(nominees)
