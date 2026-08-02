"""Unified clustering evaluation metrics.

Standard metrics: ARI, NMI, Purity, F1, Rand Index
TARImpute-specific: Q (clustering quality), Stability, DD, KNNP
Utility: compute_all_metrics() returns a dict of all metrics at once.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════
# Standard clustering metrics
# ═══════════════════════════════════════════════════════════════════════════

def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Purity: fraction of samples assigned to the majority class in each cluster.

    Purity ∈ [0, 1]. Higher is better.
    Limitation: favors many small clusters (trivially maxed at purity=1 when each
    point is its own cluster). Always pair with NMI or ARI.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = contingency_matrix(y_true, y_pred)
    return float(np.sum(np.amax(cm, axis=0)) / np.sum(cm))


def pairwise_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pairwise F1 score: treats clustering as a binary pairwise classification task.

    Two points are "in the same class" if they share the same true label and
    "in the same cluster" if they share the same predicted label. F1 is the
    harmonic mean of precision and recall on this pairwise task.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Use fast vectorized computation for moderate n
    n = len(y_true)
    if n > 5000:
        # Subsample pairs for large n
        rng = np.random.default_rng(42)
        n_pairs = min(50000, n * (n - 1) // 2)
        idx = rng.choice(n, size=(n_pairs, 2), replace=True)
        true_same = y_true[idx[:, 0]] == y_true[idx[:, 1]]
        pred_same = y_pred[idx[:, 0]] == y_pred[idx[:, 1]]
    else:
        true_eq = y_true[:, None] == y_true[None, :]
        pred_eq = y_pred[:, None] == y_pred[None, :]
        upper = np.triu_indices(n, k=1)
        true_same = true_eq[upper]
        pred_same = pred_eq[upper]

    tp = (true_same & pred_same).sum()
    fp = (~true_same & pred_same).sum()
    fn = (true_same & ~pred_same).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return float(2 * precision * recall / max(precision + recall, 1e-8))


def rand_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rand Index (RI): fraction of pairwise decisions that are correct.

    RI = (TP + TN) / (total pairs). RI ∈ [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)

    if n > 5000:
        # Subsample pairs
        rng = np.random.default_rng(42)
        n_pairs = min(50000, n * (n - 1) // 2)
        idx = rng.choice(n, size=(n_pairs, 2), replace=True)
        true_same = y_true[idx[:, 0]] == y_true[idx[:, 1]]
        pred_same = y_pred[idx[:, 0]] == y_pred[idx[:, 1]]
    else:
        true_eq = y_true[:, None] == y_true[None, :]
        pred_eq = y_pred[:, None] == y_pred[None, :]
        upper = np.triu_indices(n, k=1)
        true_same = true_eq[upper]
        pred_same = pred_eq[upper]

    tp_tn = (true_same == pred_same).sum()
    total = len(true_same)
    return float(tp_tn / max(total, 1))


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering Accuracy: best one-to-one mapping between clusters and labels.

    Uses Hungarian algorithm to find the optimal label permutation.
    Equivalent to 1 - minimum classification error after label matching.
    """
    from scipy.optimize import linear_sum_assignment

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize
    return float(cm[row_ind, col_ind].sum() / cm.sum())


# ═══════════════════════════════════════════════════════════════════════════
# TARImpute-specific structural metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_q(
    X_filled: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray | None = None,
) -> float:
    """Clustering quality Q: supervised (ARI) or unsupervised (normalized Silhouette).

    Q ∈ [0, 1]. Higher is better.
    """
    if y_true is not None:
        try:
            ari = float(adjusted_rand_score(y_true, labels))
            return float(np.clip(ari, 0.0, 1.0))
        except Exception:
            pass
    sil = silhouette_score(
        StandardScaler().fit_transform(np.nan_to_num(X_filled, nan=0.0)), labels,
    )
    if not np.isfinite(sil):
        return 0.0
    return float(np.clip((sil + 1.0) / 2.0, 0.0, 1.0))


def compute_stability(
    X_filled: np.ndarray,
    labels: np.ndarray,
    *,
    n_repeats: int = 10,
    perturbation_scale: float = 0.01,
    cluster_spec=None,
    random_state: int = 42,
) -> float:
    """Clustering stability via pairwise ARI under perturbation.

    Adds small Gaussian noise and re-clusters, then computes mean pairwise ARI.
    Stability ∈ [0, 1]. Higher = more stable.
    """
    from sklearn.metrics import adjusted_rand_score

    X = np.asarray(X_filled, dtype=float)
    scale = np.nanstd(X, axis=0)
    scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0
    rng = np.random.default_rng(random_state)

    perturbed_labels: list[np.ndarray] = [labels]
    if cluster_spec is None:
        from sklearn.cluster import KMeans
        def _cluster(X_, seed):
            return KMeans(n_clusters=len(set(labels)), n_init=5, random_state=seed).fit_predict(
                StandardScaler().fit_transform(X_),
            )
    else:
        def _cluster(X_, seed):
            return cluster_spec(X_, seed)

    for i in range(n_repeats):
        noise = rng.normal(0.0, perturbation_scale, size=X.shape) * scale
        plabels = _cluster(X + noise, random_state + i + 1)
        perturbed_labels.append(plabels)

    pairwise_aris = []
    for i in range(len(perturbed_labels)):
        for j in range(i + 1, len(perturbed_labels)):
            pairwise_aris.append(
                float(adjusted_rand_score(perturbed_labels[i], perturbed_labels[j]))
            )
    return float(np.mean(pairwise_aris)) if pairwise_aris else 0.0


def compute_dd(
    X_base: np.ndarray,
    X_filled: np.ndarray,
    n_pairs: int = 5000,
    random_state: int = 42,
) -> float:
    """Distance Distortion via Spearman rank correlation.

    DD ∈ [0, 1]. Higher = better preservation (ρ clamped to [0,1]).
    """
    n = X_base.shape[0]
    if n < 5:
        return 1.0
    rng = np.random.default_rng(random_state)
    total_pairs = n * (n - 1) // 2
    k = min(n_pairs, total_pairs)

    # Sample pair indices
    flat_idx = rng.choice(total_pairs, size=k, replace=False)
    i_idx = np.zeros(k, dtype=np.int64)
    j_idx = np.zeros(k, dtype=np.int64)
    for idx in range(k):
        flat = int(flat_idx[idx])
        i = int((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8 * flat)) // 2)
        j = int(flat - i * (2 * n - i - 1) // 2 + i + 1)
        i_idx[idx] = i
        j_idx[idx] = j

    d0 = np.sqrt(np.sum((X_base[i_idx] - X_base[j_idx]) ** 2, axis=1))
    d1 = np.sqrt(np.sum((X_filled[i_idx] - X_filled[j_idx]) ** 2, axis=1))

    rho, _ = sp_stats.spearmanr(d0, d1)
    if np.isnan(rho):
        rho_p = np.corrcoef(d0, d1)[0, 1]
        rho = float(rho_p) if not np.isnan(rho_p) else 0.5

    return float(max(0.0, min(1.0, float(rho))))


def compute_knnp(
    X_base: np.ndarray,
    X_filled: np.ndarray,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> float:
    """kNN Preservation: mean k-NN Jaccard overlap across k values.

    KNNP ∈ [0, 1]. Higher = better neighborhood preservation.
    """
    n = X_base.shape[0]
    if n <= max(k_values) + 1:
        return 1.0

    Xb = StandardScaler().fit_transform(np.nan_to_num(X_base, nan=0.0))
    Xf = StandardScaler().fit_transform(np.nan_to_num(X_filled, nan=0.0))

    scores = []
    for k in k_values:
        k_eff = min(k + 1, n)
        nn_base = NearestNeighbors(n_neighbors=k_eff).fit(Xb).kneighbors(return_distance=False)
        nn_fill = NearestNeighbors(n_neighbors=k_eff).fit(Xf).kneighbors(return_distance=False)

        overlaps = []
        for nb, nf in zip(nn_base, nn_fill, strict=True):
            s_base = set(int(x) for x in nb[1:])
            s_fill = set(int(x) for x in nf[1:])
            denom = max(1, len(s_base))
            overlaps.append(len(s_base & s_fill) / denom)
        scores.append(float(np.mean(overlaps)))

    return float(np.mean(scores))


def compute_objective(
    *,
    q: float,
    stability: float,
    knnp: float,
    dd: float,
    weights: dict[str, float] | None = None,
) -> float:
    """TARImpute composite objective.

    Default weights: q=0.30, stability=0.35, knnp=0.25, dd=0.10
    """
    w = weights or {"q": 0.30, "stability": 0.35, "knnp": 0.25, "dd": 0.10}
    return float(
        w["q"] * q + w["stability"] * stability +
        w["knnp"] * knnp + w["dd"] * dd
    )


# ═══════════════════════════════════════════════════════════════════════════
# Unified evaluation interface
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_clustering(
    X_filled: np.ndarray,
    labels: np.ndarray,
    y_true: np.ndarray,
    X_base: np.ndarray | None = None,
    *,
    compute_structural: bool = False,
    stability_repeats: int = 10,
    structural_kwargs: dict | None = None,
) -> dict[str, float]:
    """Compute all clustering metrics in one call.

    Parameters
    ----------
    X_filled : ndarray
        Imputed data matrix.
    labels : ndarray
        Cluster labels.
    y_true : ndarray
        Ground-truth labels (for external metrics).
    X_base : ndarray, optional
        Reference matrix for structural metrics (DD, KNNP).
    compute_structural : bool
        If True, also compute DD, KNNP, Stability.
    stability_repeats : int
        Number of perturbation repeats for stability.
    structural_kwargs : dict, optional
        Extra kwargs passed to structural metric functions.

    Returns
    -------
    dict with keys: ari, nmi, purity, f1, ri, accuracy, silhouette,
                    calinski_harabasz, davies_bouldin,
                    [q, stability, dd, knnp, objective]
    """
    metrics: dict[str, float] = {}

    # External metrics (require ground truth)
    metrics["ari"] = float(adjusted_rand_score(y_true, labels))
    metrics["nmi"] = float(normalized_mutual_info_score(y_true, labels))
    metrics["purity"] = purity_score(y_true, labels)
    metrics["f1"] = pairwise_f1_score(y_true, labels)
    metrics["ri"] = rand_index(y_true, labels)
    metrics["accuracy"] = clustering_accuracy(y_true, labels)

    # Internal metrics (no ground truth needed)
    X_s = StandardScaler().fit_transform(np.nan_to_num(X_filled, nan=0.0))
    metrics["silhouette"] = float(silhouette_score(X_s, labels))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_s, labels))
    metrics["davies_bouldin"] = float(davies_bouldin_score(X_s, labels))

    # TARImpute structural metrics
    if compute_structural:
        kwargs = structural_kwargs or {}
        metrics["q"] = compute_q(X_filled, labels, y_true)
        metrics["stability"] = compute_stability(
            X_filled, labels, n_repeats=stability_repeats,
            random_state=kwargs.get("random_state", 42),
        )
        if X_base is not None:
            metrics["dd"] = compute_dd(
                X_base, X_filled,
                n_pairs=kwargs.get("dd_n_pairs", 5000),
                random_state=kwargs.get("random_state", 42),
            )
            metrics["knnp"] = compute_knnp(X_base, X_filled)
            metrics["objective"] = compute_objective(
                q=metrics.get("q", metrics["silhouette"]),
                stability=metrics["stability"],
                knnp=metrics["knnp"],
                dd=metrics.get("dd", 1.0),
                weights=kwargs.get("weights"),
            )

    return metrics


# Convenience: metric names for LaTeX tables
METRIC_DISPLAY_NAMES = {
    "ari": "ARI",
    "nmi": "NMI",
    "purity": "Purity",
    "f1": "F1",
    "ri": "RI",
    "accuracy": "ACC",
    "silhouette": "Silhouette",
    "calinski_harabasz": "CH",
    "davies_bouldin": "DBI",
    "q": "Q",
    "stability": "Stability",
    "dd": "DD",
    "knnp": "kNNP",
    "objective": "Obj",
}
