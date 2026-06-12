"""
enterprise_gnn_model.py — GraphSAGE model for Enterprise RCA GNN.

Call check_torch_geo_requirement() in scripts before using model functions.
No remediation content is produced here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .enterprise_gnn_dataset import IN_DIM, FEATURE_NAMES

# ── Lazy dependency detection ───────────────────────────────────────────────────

_torch_available = False
_pyg_available   = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _torch_available = True
    try:
        from torch_geometric.nn import SAGEConv
        _pyg_available = True
    except ImportError:
        pass
except ImportError:
    pass


# ── Dependency check ────────────────────────────────────────────────────────────

def check_torch_geo_requirement() -> None:
    """Exit with a clear message if torch or torch_geometric is missing."""
    if not _torch_available:
        print("[ERROR] PyTorch is required for enterprise GNN RCA.")
        print("        Install from: https://pytorch.org/get-started/locally/")
        sys.exit(1)
    if not _pyg_available:
        print("[ERROR] torch_geometric is required for enterprise GNN RCA.")
        print("        Install with the correct torch/ROCm/CUDA wheel.")
        print("        See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
        sys.exit(1)


# ── Model ─────────────────────────────────────────────────────────────────────
# Defined unconditionally so top-level imports always succeed.
# When deps are missing, instantiation raises RuntimeError (scripts call
# check_torch_geo_requirement() first which exits before reaching __init__).

if _pyg_available:
    class EnterpriseRcaGNN(nn.Module):
        """
        3-layer GraphSAGE network for node-level root-cause scoring.

        Output: one scalar logit per node.  Apply softmax at inference for
        normalised confidence scores.
        """

        def __init__(
            self,
            in_channels:     int   = IN_DIM,
            hidden_channels: int   = 64,
            num_layers:      int   = 3,
            dropout:         float = 0.2,
        ) -> None:
            super().__init__()
            self.in_channels     = in_channels
            self.hidden_channels = hidden_channels
            self.num_layers      = num_layers
            self.dropout_p       = dropout

            self.input_lin = nn.Linear(in_channels, hidden_channels)
            self.convs = nn.ModuleList([
                SAGEConv(hidden_channels, hidden_channels)
                for _ in range(num_layers)
            ])
            self.output_lin = nn.Linear(hidden_channels, 1)

        def forward(self, x, edge_index):
            x = F.relu(self.input_lin(x))
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
            return self.output_lin(x).squeeze(-1)  # [num_nodes]

else:
    class EnterpriseRcaGNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "EnterpriseRcaGNN requires torch_geometric. "
                "Call check_torch_geo_requirement() before instantiating the model."
            )


# ── Config helpers ─────────────────────────────────────────────────────────────

def build_gnn_config(
    in_channels:     int   = IN_DIM,
    hidden_channels: int   = 64,
    num_layers:      int   = 3,
    dropout:         float = 0.2,
    top_k:           int   = 3,
) -> dict:
    return {
        "in_channels":     in_channels,
        "hidden_channels": hidden_channels,
        "num_layers":      num_layers,
        "dropout":         dropout,
        "top_k":           top_k,
        "model_type":      "EnterpriseRcaGNN",
        "gnn_architecture": "GraphSAGE",
        "feature_dim":     IN_DIM,
    }


# ── Serialisation ──────────────────────────────────────────────────────────────

def save_gnn(
    model: "EnterpriseRcaGNN",
    config: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch as _t
    _t.save(model.state_dict(), out_dir / "enterprise_gnn_rca.pt")
    (out_dir / "enterprise_gnn_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "feature_columns.json").write_text(
        json.dumps(FEATURE_NAMES, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_gnn(model_path: Path, config_path: Path) -> tuple["EnterpriseRcaGNN", dict]:
    import torch as _t
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model  = EnterpriseRcaGNN(
        in_channels=config.get("in_channels", IN_DIM),
        hidden_channels=config.get("hidden_channels", 64),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.2),
    )
    model.load_state_dict(_t.load(str(model_path), map_location="cpu"))
    model.eval()
    return model, config


# ── PyG Data conversion ────────────────────────────────────────────────────────

def graph_dict_to_pyg(g: dict):
    """Convert a build-script graph_dict to a PyG Data object."""
    from torch_geometric.data import Data
    return Data(
        x=g["x"],
        edge_index=g["edge_index"],
        y=g["y"],
        num_nodes=g["num_nodes"],
    )


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_nodes(model: "EnterpriseRcaGNN", data) -> np.ndarray:
    """Run model and return softmax-normalised per-node probabilities."""
    import torch as _t
    model.eval()
    with _t.no_grad():
        logits = model(data.x, data.edge_index)
        probs  = _t.softmax(logits, dim=0).cpu().numpy()
    return probs
