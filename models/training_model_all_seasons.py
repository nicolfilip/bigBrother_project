import pandas as pd
import numpy as np
import networkx as nx
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/big_brother_israel_new_new.csv")
df["full_name"] = df["שם מלא"].astype(str)
df["week_eliminated"] = pd.to_numeric(df["week eliminated"], errors="coerce")
df["entered_week"] = pd.to_numeric(df["entered week"], errors="coerce")
df["rank_score"] = df["week_eliminated"] - df["entered_week"]
df["season"] = pd.to_numeric(df["עונה"], errors="coerce")
df["gender_encoded"] = df["מין"].map({"ז": 1, "נ": 0})
df["status_encoded"] = df["סטטוס"].astype(str).apply(lambda s: 1 if "נשוי" in s else 0)
df["is_vip"] = df["שם מלא"].astype(str).apply(lambda x: 1 if "VIP" in x or "מהעונה" in x else 0)
df["age"] = pd.to_numeric(df["גיל"], errors="coerce")

# Base features
base_features = ["age", "gender_encoded", "status_encoded", "is_vip"]

# Graph features for Season 12 only
edges_df = pd.read_csv("../graph_output/season12/graph_eng_12.csv")
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
    s = d.get("sentiment", "neutral")
    w = d.get("weight", 1)
    sentiment_weights[s][u] = sentiment_weights[s].get(u, 0) + w

# Add graph features only to Season 12
df_graph = df[df["season"] == 12].copy()
df_graph = df_graph.assign(
    pagerank=df_graph["full_name"].map(pagerank).fillna(0),
    betweenness=df_graph["full_name"].map(betweenness).fillna(0),
    closeness=df_graph["full_name"].map(closeness).fillna(0),
    in_degree=df_graph["full_name"].map(in_degree).fillna(0),
    out_degree=df_graph["full_name"].map(out_degree).fillna(0),
    total_degree=df_graph["full_name"].map(total_degree).fillna(0),
    **{f"{s}_out_weight": df_graph["full_name"].map(sentiment_weights[s]).fillna(0) for s in sentiment_weights}
)

graph_features = [
    "pagerank", "betweenness", "closeness", "in_degree", "out_degree", "total_degree",
    "positive_out_weight", "negative_out_weight", "romantic_out_weight", "neutral_out_weight"
]


# Function for dynamic weekly training
def dynamic_weekly_training(df_data, features, title):
    ndcg_scores = []
    correct_preds = 0
    total_preds = 0
    predictions = []

    for week in range(2, int(df_data["week_eliminated"].max()) + 1):
        train = df_data[df_data["week_eliminated"] < week]
        test = df_data[df_data["week_eliminated"] == week]

        if train.empty or len(test) < 2:
            continue

        X_train, y_train = train[features], train["rank_score"]
        X_test, y_test = test[features], test["rank_score"]

        model = XGBRanker(objective="rank:pairwise", n_estimators=25, random_state=42)
        model.fit(X_train, y_train, group=[len(X_train)])
        preds = model.predict(X_test)

        ndcg = ndcg_score([y_test.values], [preds])
        ndcg_scores.append(ndcg)

        predicted = test.iloc[np.argmin(preds)]["full_name"]
        actual = test.iloc[np.argmin(y_test.values)]["full_name"]
        predictions.append((week, predicted, actual))

        if predicted == actual:
            correct_preds += 1
        total_preds += 1

    print(f"\nResults for {title}:")
    for week, pred, actual in predictions:
        print(f"Week {week}: {pred} (Actual: {actual}) {'✅' if pred == actual else '❌'}")

    print(f"Accuracy: {correct_preds}/{total_preds} = {correct_preds/total_preds:.2%}")
    print(f"Average nDCG: {np.mean(ndcg_scores):.4f}")


# Run dynamic training for all seasons except 15 (assuming 15 is live)
df_all = df[df["season"] <= 14].copy()
dynamic_weekly_training(df_all, base_features, "All Seasons Until Season 14")

# Run dynamic training for Season 12 only with graph features
all_features_12 = base_features + graph_features
dynamic_weekly_training(df_graph, all_features_12, "Season 12 Only (with Graph)")
