import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ------------------------
# 7. גרף קשרים ידני מקובץ CSV
# ------------------------

# טען את קובץ הקשרים הידני שיצרת
manual_edges = pd.read_csv("Untitled spreadsheet - Sheet1.csv")

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
plt.savefig("bigbrother_manual_graph.png", dpi=300)
plt.show()

print("גרף שמור כקובץ: graph_output/bigbrother_manual_graph.png")

