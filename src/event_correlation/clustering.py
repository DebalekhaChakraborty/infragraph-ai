"""
clustering.py — Graph-aware alert clustering.

cluster_alerts(alerts, features, threshold) -> dict
    Uses sklearn AgglomerativeClustering when available,
    falls back to union-find graph-connected-component clustering.
"""
from __future__ import annotations

from .features import compute_pairwise_similarity

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

try:
    from sklearn.cluster import AgglomerativeClustering  # type: ignore
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def _build_similarity_matrix(features: list[dict]) -> list[list[float]]:
    n = len(features)
    mat: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                mat[i][j] = 1.0
            else:
                s = compute_pairwise_similarity(features[i], features[j])
                mat[i][j] = s
                mat[j][i] = s
    return mat


def _union_find_cluster(features: list[dict], threshold: float) -> list[int]:
    """
    Greedy union-find: merge alerts if pairwise similarity >= threshold.
    O(n^2) — acceptable for n <= 500.
    """
    n = len(features)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if compute_pairwise_similarity(features[i], features[j]) >= threshold:
                union(i, j)

    roots: dict[int, int] = {}
    labels: list[int] = []
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels.append(roots[r])
    return labels


def cluster_alerts(
    alerts: list[dict],
    features: list[dict],
    threshold: float = 0.62,
) -> dict:
    """
    Cluster alerts by AI/graph-aware similarity.

    Returns
    -------
    dict with:
      labels      : list[int] — cluster index per alert
      n_clusters  : int
      method      : str
      similarity_matrix : list[list[float]]  (only when n <= 30)
    """
    n = len(alerts)
    if n == 0:
        return {"labels": [], "n_clusters": 0, "method": "empty"}
    if n == 1:
        return {"labels": [0], "n_clusters": 1, "method": "singleton"}

    method = "graph_connected_components"
    labels: list[int]

    if _SKLEARN and _NP and n <= 300:
        try:
            mat = _build_similarity_matrix(features)
            dist_mat = np.array([[1.0 - mat[i][j] for j in range(n)] for i in range(n)])
            clust = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.0 - threshold,
                linkage="average",
                metric="precomputed",
            )
            clust.fit(dist_mat)
            labels = [int(lb) for lb in clust.labels_]
            method = "sklearn_agglomerative"
        except Exception:
            labels = _union_find_cluster(features, threshold)
    else:
        labels = _union_find_cluster(features, threshold)

    n_clusters = len(set(labels))
    result: dict = {"labels": labels, "n_clusters": n_clusters, "method": method}
    if n <= 30:
        result["similarity_matrix"] = _build_similarity_matrix(features)
    return result
