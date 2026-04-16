# src/graph_builder.py
# Build a misinformation spread network using NetworkX

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import json

os.makedirs("outputs/graphs", exist_ok=True)

print("Loading cleaned data...")
df = pd.read_csv("data/processed/fakenews_clean.csv")

# Use a sample to keep graph manageable
df_sample = df.sample(n=500, random_state=42).reset_index(drop=True)

# ── Build Directed Graph ─────────────────────────────────────
print("Building spread network...")
G = nx.DiGraph()

np.random.seed(42)

for idx, row in df_sample.iterrows():
    source = f"source_{idx}"
    label  = row["label_name"]

    # Add source node
    G.add_node(source, label=label, type="source")

    # Fake news spreads to more nodes (super spreaders)
    n_spreaders = np.random.randint(3, 10) if label == "fake" else np.random.randint(1, 5)

    for i in range(n_spreaders):
        spreader = f"user_{idx}_{i}"
        G.add_node(spreader, label=label, type="spreader")
        G.add_edge(source, spreader)

        # Second level spread (fake spreads further)
        n_secondary = np.random.randint(1, 5) if label == "fake" else np.random.randint(0, 2)
        for j in range(n_secondary):
            secondary = f"user_{idx}_{i}_{j}"
            G.add_node(secondary, label=label, type="secondary")
            G.add_edge(spreader, secondary)

print(f"Graph built → Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

# ── Graph Metrics ────────────────────────────────────────────
print("\nCalculating network metrics...")

density         = nx.density(G)
num_nodes       = G.number_of_nodes()
num_edges       = G.number_of_edges()

# In-degree centrality (who receives the most retweets)
in_degree       = nx.in_degree_centrality(G)
out_degree      = nx.out_degree_centrality(G)

# Top 5 super spreader nodes
top_spreaders = sorted(out_degree.items(), key=lambda x: -x[1])[:5]

print(f"\n📊 Network Metrics:")
print(f"   Nodes        : {num_nodes}")
print(f"   Edges        : {num_edges}")
print(f"   Density      : {density:.6f}")
print(f"\n🔥 Top 5 Super Spreaders:")
for node, score in top_spreaders:
    print(f"   {node} → centrality: {score:.4f}")

# Save metrics to JSON
metrics = {
    "nodes"      : num_nodes,
    "edges"      : num_edges,
    "density"    : density,
    "top_spreaders": [{"node": n, "score": round(s, 4)} for n, s in top_spreaders]
}
with open("outputs/graphs/network_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("\n✅ Saved network_metrics.json")

# ── Plot 1: Spread Network (sample of 80 nodes) ──────────────
print("\nPlotting network graph...")
subgraph_nodes  = list(G.nodes())[:80]
H               = G.subgraph(subgraph_nodes)

node_colors = []
for node in H.nodes():
    lbl = G.nodes[node].get("label", "real")
    node_colors.append("#e74c3c" if lbl == "fake" else "#2ecc71")

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(H, seed=42, k=0.5)
nx.draw_networkx(
    H, pos,
    node_color=node_colors,
    node_size=100,
    edge_color="#aaaaaa",
    arrows=True,
    with_labels=False,
    alpha=0.8
)
from matplotlib.patches import Patch
legend = [Patch(color="#e74c3c", label="Fake"), Patch(color="#2ecc71", label="Real")]
plt.legend(handles=legend, fontsize=12)
plt.title("Misinformation Spread Network\n(Red = Fake, Green = Real)", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/graphs/spread_network.png", dpi=150)
plt.show()
print("✅ Saved spread_network.png")

# ── Plot 2: Fake vs Real — Node Count in Graph ───────────────
fake_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "fake")
real_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("label") == "real")

plt.figure(figsize=(6,4))
plt.bar(["Fake", "Real"], [fake_nodes, real_nodes],
        color=["#e74c3c", "#2ecc71"], edgecolor="black")
plt.title("Fake vs Real Nodes in Spread Network")
plt.ylabel("Number of Nodes")
plt.tight_layout()
plt.savefig("outputs/graphs/network_node_distribution.png", dpi=150)
plt.show()
print("✅ Saved network_node_distribution.png")

# ── Export graph for Gephi ───────────────────────────────────
nx.write_gexf(G, "outputs/graphs/spread_network.gexf")
print("✅ Saved spread_network.gexf (open in Gephi for visualization)")

print("\n✅ Network analysis complete!")