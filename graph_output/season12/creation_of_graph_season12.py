import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

season_12_edges = pd.read_csv("graph_eng_12.csv")
G = nx.DiGraph()
for _, row in season_12_edges.iterrows():
    G.add_edge(
        row["from"].strip(),
        row["to"].strip(),
        weight=float(row["weight"]),
        sentiment=row["sentiment"].strip().lower()
    )


def get_edge_color(sentiment):
    if sentiment == "positive":
        return "green"
    elif sentiment == "negative":
        return "red"
    elif sentiment == "romantic":
        return "hotpink"
    else:
        return "orange"  # ניטרלי או לא מוגדר

edge_colors = [get_edge_color(G[u][v].get("sentiment", "neutral")) for u, v in G.edges()]

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(16, 10))
nx.draw(
    G, pos,
    with_labels=True,
    edge_color=edge_colors,
    node_color="lightblue",
    node_size=1500,
    font_size=10,
    width=2,
    arrows=True,
    arrowstyle='->',
    arrowsize=20
)

plt.title("גרף קשרים ידני – עונה 12")
plt.savefig("bigbrother_manual_graph_season_12.png", dpi=300)
plt.show()

print("✅ גרף שמור כקובץ: bigbrother_manual_graph_season_12.png")
