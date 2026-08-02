"""Reusable clustering evaluation helpers for clustering-oriented imputation studies."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
    KMeans,
    SpectralClustering,
)
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ClustererName = Literal["kmeans", "gmm", "dbscan", "hdbscan", "spectral", "agglomerative"]


def cluster_and_score(
    X: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
    algorithm: ClustererName = "kmeans",
    dbscan_kwargs: dict | None = None,
    hdbscan_kwargs: dict | None = None,
    agglomerative_kwargs: dict | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Cluster the input matrix and compute internal clustering indices.

    Supports KMeans, GMM, DBSCAN, HDBSCAN, SpectralClustering, Agglomerative.
    """
    X = np.asarray(X, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "dbscan":
        db_kwargs = dbscan_kwargs or {}
        eps = db_kwargs.get("eps", _estimate_dbscan_eps(X_scaled))
        min_samples = db_kwargs.get("min_samples", 5)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    elif algorithm == "hdbscan":
        hdb_kwargs = hdbscan_kwargs or {}
        min_cluster_size = hdb_kwargs.get("min_cluster_size", 5)
        min_samples = hdb_kwargs.get("min_samples", max(1, min_cluster_size // 2))
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
        ).fit_predict(X_scaled)
    elif algorithm == "spectral":
        labels = SpectralClustering(
            n_clusters=n_clusters,
            affinity="rbf",
            random_state=random_state,
            assign_labels="kmeans",
        ).fit_predict(X_scaled)
    elif algorithm == "agglomerative":
        agg_kwargs = agglomerative_kwargs or {}
        linkage = agg_kwargs.get("linkage", "ward")
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage,
        ).fit_predict(X_scaled)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    metrics = {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
    }
    return labels, metrics


def _estimate_dbscan_eps(X_scaled: np.ndarray, min_samples: int = 5) -> float:
    """自适应估计DBSCAN eps参数（k-distance elbow法）。"""
    n = X_scaled.shape[0]
    k = min(min_samples, n - 1)
    if k < 1:
        return 0.5
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_dists = np.sort(distances[:, k])
    x = np.arange(len(k_dists))
    y = k_dists
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return float(k_dists[-1] * 0.5)
    line_unit = line_vec / line_len
    point_vecs = np.column_stack([x - x[0], y - y[0]])
    cross = np.abs(point_vecs[:, 0] * line_unit[1] - point_vecs[:, 1] * line_unit[0])
    elbow_idx = int(np.argmax(cross))
    window = k_dists[max(0, elbow_idx - 2):min(len(k_dists), elbow_idx + 3)]
    eps = float(np.median(window))
    return max(0.05, min(eps, k_dists[-1] * 0.8))
