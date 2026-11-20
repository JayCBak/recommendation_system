import numpy as np


def baseline(R):
    # Calculate overall, user, and movie effects
    known = ~np.isnan(R)
    mu = np.nanmean(R)

    b_user = np.nanmean(R - mu, axis=1)
    residual = R - mu - b_user[:, None]

    n_item = known.sum(axis=0)
    b_item = np.zeros(R.shape[1])
    has_rating = n_item > 0

    b_item[has_rating] = (
        np.nansum(residual, axis=0)[has_rating]
        / n_item[has_rating]
    )

    return mu, b_user, b_item


# Milestone 2 name for the same function
bias_baseline = baseline


def baseline_matrix(R):
    """Full matrix of baseline predictions mu + b_user + b_item."""
    mu, b_user, b_item = baseline(R)
    return mu + b_user[:, None] + b_item[None, :]


def rank_k(U, s, Vt, k):
    """Rank-k matrix from an SVD that was already computed."""
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def iterative_svd_predict(R, k, n_iter=15):
    """Model 2, then repeatedly refill the gaps with the rank-k approximation."""
    known = ~np.isnan(R)
    B = baseline_matrix(R)
    X = np.where(known, R - B, 0.0)
    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = rank_k(U, s, Vt, k)
        X = np.where(known, R - B, L)  # known ratings stay fixed
    return L + B
