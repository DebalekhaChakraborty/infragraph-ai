"""
GNN-based Root Cause Analysis using PyTorch Geometric.

A lightweight Graph Attention Network (GAT) that learns to predict the
root-cause node from alert-annotated topology graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

# Class → integer feature index
_CLASS_IDX = {
    "router": 0, "switch": 1, "firewall": 2, "server": 3,
    "database": 4, "load_balancer": 5, "cloud_or_wan": 6,
}
_N_CLASSES  = len(_CLASS_IDX)
_SEVERITY_F = {"critical": 1.0, "major": 0.67, "warning": 0.33, "none": 0.0}


class GATRCAModel(nn.Module):
    """Two-layer Graph Attention Network for node-level root-cause prediction."""

    def __init__(self, in_channels: int = _N_CLASSES + 2, hidden: int = 32, heads: int = 4):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise ImportError("torch_geometric is required. Run: pip install torch-geometric")
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden * heads, 1,   heads=1,     concat=False)

    def forward(self, data: "Data") -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)  # (N,) logits


def graph_to_pyg(G, alerts: list[dict]) -> "Data":
    """Convert a NetworkX topology graph + alert list to a PyG Data object.

    Node features: one-hot class (7) + alert severity (1) + is_alerted (1).
    """
    if not _PYG_AVAILABLE:
        raise ImportError("torch_geometric is required.")

    import networkx as nx

    node_list = list(G.nodes())
    idx = {n: i for i, n in enumerate(node_list)}

    # Build alert lookup
    sev_map: dict[str, float] = {}
    for a in alerts:
        n = a.get("node")
        if n in idx:
            sev_map[n] = max(sev_map.get(n, 0.0), _SEVERITY_F.get(a.get("severity", "none"), 0.0))

    feats = []
    for n in node_list:
        cls   = G.nodes[n].get("class_name", "server")
        oh    = [0.0] * _N_CLASSES
        oh[_CLASS_IDX.get(cls, 3)] = 1.0
        sev   = sev_map.get(n, 0.0)
        alerted = 1.0 if n in sev_map else 0.0
        feats.append(oh + [sev, alerted])

    x = torch.tensor(feats, dtype=torch.float)

    edges = [(idx[u], idx[v]) for u, v in G.edges() if u in idx and v in idx]
    if edges:
        src, dst = zip(*edges)
        # Undirected: add both directions
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def predict_root_cause(
    model: GATRCAModel,
    G,
    alerts: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Run the trained GNN and return top-k root-cause candidates.

    Returns list of dicts ``{node, class_name, score}`` sorted descending.
    """
    model.eval()
    data   = graph_to_pyg(G, alerts)
    node_list = list(G.nodes())
    with torch.no_grad():
        logits = model(data)
        probs  = torch.softmax(logits, dim=0).cpu().numpy()

    ranked = sorted(
        [{"node": n, "class_name": G.nodes[n].get("class_name", "?"),
          "score": float(probs[i])}
         for i, n in enumerate(node_list)],
        key=lambda x: x["score"], reverse=True,
    )
    return ranked[:top_k]
