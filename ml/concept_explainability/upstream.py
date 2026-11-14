"""Numerical routines copied from Adam Kukucka's IBD concept workflow.

RationAI/ulcerative-colitis @ a6fe08b93219b91a68758a4eaf32159a33f5d4eb.
Numerical function bodies are unchanged; see third_party/ibd-explainability-NOTICE.txt.
"""
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ray.data
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


UPSTREAM_SHA = "a6fe08b93219b91a68758a4eaf32159a33f5d4eb"

def nnls_batch(
    x: np.ndarray,
    h: np.ndarray,
    n_iter: int,
    w0: np.ndarray | None = None,
    tol: float | None = None,
) -> np.ndarray:
    """Solve `min_W ||X - W H||_F^2  s.t. W >= 0` for every row of X at once.

    Accelerated projected gradient (FISTA) with step 1/L, L the largest
    eigenvalue of `H H^T` - each iteration is only `(n, K) @ (K, K)`, so
    hundreds of iterations are cheap at small K. Starts from `w0` if given,
    else from the clipped unconstrained least-squares solution.

    Args:
        x: Rows to fit, shape `(n, d)`.
        h: Fixed dictionary, shape `(k, d)` - any sign.
        n_iter: Number of FISTA iterations (the maximum, if `tol` is set).
        w0: Optional warm start, shape `(n, k)`.
        tol: Stop early once an iteration changes W by less than `tol`
            relative to its norm. None (the default) always runs `n_iter` -
            kept for every transform/encoder pass, so saved W stays exactly
            reproducible.

    Returns:
        W, shape `(n, k)`, non-negative.
    """
    gram = h @ h.T
    xh = x @ h.T
    lipschitz = float(np.linalg.eigvalsh(gram).max())
    if lipschitz <= 0:
        return np.zeros_like(xh)
    if w0 is None:
        w0 = np.maximum(np.linalg.lstsq(gram, xh.T, rcond=None)[0].T, 0.0)
    w = w0.astype(xh.dtype, copy=True)
    y = w.copy()
    t = 1.0
    for _ in range(n_iter):
        w_next = np.maximum(y - (y @ gram - xh) / lipschitz, 0.0)
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = w_next + ((t - 1.0) / t_next) * (w_next - w)
        converged = tol is not None and np.linalg.norm(w_next - w) <= tol * max(
            float(np.linalg.norm(w)), 1e-12
        )
        w, t = w_next, t_next
        if converged:
            break
    return w

def kmeans_init(x: np.ndarray, k: int, random_state: int, max_rows: int = 20_000) -> np.ndarray:
    """Initial dictionary from k-means centroids on (at most `max_rows` of) `x`.

    Ding et al.'s recommended semi-NMF initialization - semi-NMF is a relaxed
    soft k-means, so k-means centroids are already a good fixed point to
    start from.
    """
    rng = np.random.default_rng(random_state)
    sample = x if len(x) <= max_rows else x[rng.choice(len(x), max_rows, replace=False)]
    kmeans = KMeans(n_clusters=k, n_init=1, max_iter=100, random_state=random_state)
    # Single-threaded on purpose: multithreaded KMeans sums partial results in
    # varying order, so identical data gave slightly different centroids - and
    # every tree split downstream drifted apart. At <= max_rows rows this is cheap.
    with threadpool_limits(limits=1):
        return kmeans.fit(sample).cluster_centers_.astype(x.dtype)

def reseed_dead_components(
    h: np.ndarray, usage: np.ndarray, x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Replace dictionary rows no sample uses (`usage == 0`) with random rows of `x`."""
    dead = np.flatnonzero(usage <= 0)
    if len(dead):
        h = h.copy()
        h[dead] = x[rng.choice(len(x), len(dead), replace=False)]
    return h

def iter_embedding_batches(
    dataset: ray.data.Dataset,
    batch_size: int,
    shift: np.ndarray,
    scale: np.ndarray,
    metadata_columns: tuple[str, ...] = (),
    shuffle_seed: int | None = None,
    shuffle_buffer_size: int | None = None,
    clip: bool = True,
    transform: np.ndarray | None = None,
) -> Iterator[tuple[np.ndarray, pd.DataFrame | None]]:
    """Yield shifted, scaled (and by default non-negative) embedding batches with optional provenance.

    Args:
        dataset: `ray.data.Dataset` with an `embedding` column, in whatever
            order the caller wants read (see `shuffle_seed`).
        batch_size: Number of rows to read per yielded batch.
        shift: Per-dimension shift constant `c`, shape (embed_dim,).
        scale: Per-dimension scale constant `d` (the IQR, see `load_scale`),
            shape (embed_dim,) - matches concept_mil.tex's non-negativity
            transform t~ = (t + c) / d (here `shift` plays the role of `-c`).
        metadata_columns: Columns (e.g. `("slide_id", "x", "y")`) to also
            yield as a DataFrame aligned with each batch, so each row of W
            can be traced back to where it came from. Empty -> no metadata.
        shuffle_seed: If given, rows are read in a locally-shuffled order
            (a cheap, per-worker approximate shuffle - see `Dataset.iter_batches`'s
            `local_shuffle_buffer_size`, no cross-node data movement) - used
            for the per-epoch NMF training passes. Leave as None (read order
            preserved) for the final transform pass, since its output rows
            must line up 1:1 with the yielded metadata.
        shuffle_buffer_size: Row buffer size for the local shuffle; required
            together with `shuffle_seed`, ignored otherwise.
        clip: Clip negatives to zero (NMF's non-negativity transform). Off
            for semi-NMF, which factors the signed embeddings directly.
        transform: The output metric's T (`metric_transform`), applied last -
            rows come out as T (z - shift) / scale. None leaves them as is.

    Yields:
        Tuples of (rows, metadata), where rows has shape
        (n_rows_in_batch, embed_dim) and metadata is None unless
        `metadata_columns` is non-empty.
    """
    for batch in dataset.select_columns([*metadata_columns, "embedding"]).iter_batches(
        batch_size=batch_size,
        batch_format="numpy",
        local_shuffle_seed=shuffle_seed,
        local_shuffle_buffer_size=shuffle_buffer_size,
    ):
        raw = np.stack(batch["embedding"]).astype(np.float32, copy=False)
        rows = (raw - shift) / scale
        if clip:
            rows = np.maximum(rows, 0.0)
        if transform is not None:
            rows = rows @ transform.T

        metadata = None
        if metadata_columns:
            metadata = pd.DataFrame({column: batch[column] for column in metadata_columns})
        yield rows, metadata

def fit_semi_nmf_online(
    dataset: ray.data.Dataset,
    config: DictConfig,
    shift: np.ndarray,
    scale: np.ndarray,
    transform: np.ndarray | None = None,
) -> np.ndarray:
    """Online semi-NMF (W >= 0, H signed) over shuffled passes of the shifted, scaled, signed embeddings.

    Online dictionary learning (Mairal et al. 2010) with semi-NMF's
    unconstrained H-step: per batch, W_b is the exact NNLS solution against
    the current H, and H is re-solved from exponentially-forgotten
    sufficient statistics `A = sum W_b^T W_b`, `B = sum W_b^T X_b` -
    `H = (A + ridge I)^-1 B`. `forget` < 1 down-weights batches seen with an
    early, worse H. H is initialized from k-means centroids of the first
    batch (Ding et al.'s recommended semi-NMF init).

    Returns:
        H in the scaled (and with `transform`, T-mapped) fit space, shape
        `(n_components, embed_dim)`.
    """
    rng = np.random.default_rng(config.nmf.random_state)
    k = config.n_components
    h = a = b = None
    for epoch in range(config.nmf.epochs):
        for rows, _ in iter_embedding_batches(
            dataset,
            config.nmf.batch_size,
            shift,
            scale,
            shuffle_seed=config.nmf.random_state + epoch,
            shuffle_buffer_size=config.nmf.shuffle_buffer_size,
            clip=False,
            transform=transform,
        ):
            if h is None:
                h = kmeans_init(rows, k, config.nmf.random_state)
                a = np.zeros((k, k), dtype=np.float64)
                b = np.zeros((k, rows.shape[1]), dtype=np.float64)
            assert a is not None and b is not None
            w = nnls_batch(rows, h, config.nmf.nnls_iter)
            a = config.semi_nmf.forget * a + w.T @ w
            b = config.semi_nmf.forget * b + w.T @ rows
            h = np.linalg.solve(a + config.semi_nmf.ridge * np.eye(k), b).astype(np.float32)
            h = reseed_dead_components(h, np.diag(a), rows, rng)
    if h is None:
        raise ValueError("Dataset is empty - nothing to fit.")
    return h

def gauge_fix_dictionary(h: np.ndarray) -> np.ndarray:
    """Fix the WH scale ambiguity: rescale H to unit rows.

    For any positive diagonal S, W @ H == (W @ S^-1) @ (S @ H), so component
    magnitudes carry no meaning until this is fixed. No corresponding W
    correction is needed from the caller: `main` re-transforms every row
    against this gauge-fixed H directly rather than computing W against the
    pre-fix H and rescaling it after the fact.

    Args:
        h: Dictionary/components matrix, shape (n_components, n_features).

    Returns:
        The gauge-fixed H (unit L2-norm rows).
    """
    norms = np.linalg.norm(h, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    return h / norms[:, None]

@dataclass
class ConceptAutoencoder:
    dictionary: np.ndarray  # (K, D), rows d_k
    mu: np.ndarray  # (D,)
    clip: bool  # NMF's non-negativity clip before encoding
    nnls_iter: int
    transform: np.ndarray | None = None  # (D, D) T of a fit under the heads' output metric

    def encode(self, z: np.ndarray) -> np.ndarray:
        """V = E(z), shape `(n, K)` - NNLS in the metric the dictionary was fit in."""
        x = z - self.mu
        if self.clip:
            x = np.maximum(x, 0.0)
        atoms = self.dictionary
        if self.transform is not None:
            x, atoms = x @ self.transform.T, atoms @ self.transform.T
        return nnls_batch(x, atoms, self.nnls_iter)

    def decode(self, v: np.ndarray) -> np.ndarray:
        """D(v) = mu + v @ dictionary, shape `(n, D)`."""
        return self.mu + v @ self.dictionary

def code_share(v: np.ndarray) -> np.ndarray:
    """Each concept's share of the tile's total code, v_k / sum_l v_l, shape `v.shape`."""
    return v / np.maximum(v.sum(axis=1, keepdims=True), 1e-12)
