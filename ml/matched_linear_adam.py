"""CPU Adam for the same centered, penalized training objective as ridge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg.blas import dgemv


if TYPE_CHECKING:
    from numpy.typing import NDArray


type FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AdamFit:
    weights: FloatArray
    intercept: float
    objective: float
    steps: int
    converged: bool


def fit_matched_adam(
    x: FloatArray,
    y: FloatArray,
    alpha: float,
    optimum: float,
    *,
    max_steps: int = 2000,
    lr: float = 0.01,
    relative_tolerance: float = 1e-8,
) -> AdamFit:
    """Fit full-batch Adam against ``(SSE + alpha * sum(weights**2)) / n``.

    Centering profiles out the unpenalized intercept. The cosine schedule
    matches the archived comparison, including the minimum rate of ``lr / 1000``.
    The stopping check uses only training data and the supplied ridge optimum.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y) or not all(x.shape):
        raise ValueError("Expected a nonempty feature matrix and aligned target vector")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Expected finite training features and targets")
    if not 0 < alpha < math.inf or not 0 <= optimum < math.inf:
        raise ValueError("Expected positive alpha and a nonnegative finite optimum")
    if max_steps < 1 or not 0 < lr < math.inf or not 0 <= relative_tolerance < math.inf:
        raise ValueError(
            "Expected a positive step budget and rate and a finite tolerance"
        )

    step = 0
    try:
        with np.errstate(over="raise", invalid="raise"):
            x_mean, y_mean = x.mean(axis=0), y.mean()
            # Keep one BLAS-compatible buffer for the two matrix-vector products.
            matrix = np.asfortranarray((x - x_mean).T)
            targets = y - y_mean
            weights = np.zeros(x.shape[1], dtype=np.float64)
            first = np.zeros_like(weights)
            second = np.zeros_like(weights)
            residual = np.empty_like(targets)
            gradient = np.empty_like(weights)

            n = len(y)
            objective = float(targets @ targets / n)
            if not math.isfinite(objective):
                raise FloatingPointError("nonfinite initial objective")
            roundoff = 64 * np.finfo(np.float64).eps * objective
            tolerance = max(relative_tolerance * optimum, roundoff)

            while True:
                if objective < optimum - roundoff:
                    raise ValueError(
                        "Adam training objective falls below the supplied optimum"
                    )
                if step == max_steps or objective - optimum <= tolerance:
                    intercept = float(y_mean - x_mean @ weights)
                    if not math.isfinite(intercept):
                        raise FloatingPointError("nonfinite intercept")
                    return AdamFit(
                        weights,
                        intercept,
                        objective,
                        step,
                        objective - optimum <= tolerance,
                    )

                updates = range(step + 1, min(step + 25, max_steps) + 1)
                for step in updates:
                    residual = dgemv(
                        1.0, matrix, weights, trans=1, y=residual, overwrite_y=1
                    )
                    residual -= targets
                    gradient = dgemv(
                        2.0 / n, matrix, residual, y=gradient, overwrite_y=1
                    )
                    gradient += (2.0 * (alpha / n)) * weights
                    first += 0.1 * (gradient - first)
                    second *= 0.999
                    second += 0.001 * np.square(gradient)
                    rate = (
                        lr / 1000
                        + (lr - lr / 1000)
                        * (1 + math.cos(math.pi * (step - 1) / max_steps))
                        / 2
                    )
                    denominator = np.sqrt(second) / math.sqrt(1 - 0.999**step) + 1e-8
                    weights -= (rate / (1 - 0.9**step)) * first / denominator

                residual = dgemv(
                    1.0, matrix, weights, trans=1, y=residual, overwrite_y=1
                )
                residual -= targets
                objective = float(
                    (residual @ residual) / n + (alpha / n) * (weights @ weights)
                )
                if not math.isfinite(objective):
                    raise FloatingPointError("nonfinite training objective")
    except FloatingPointError as error:
        raise FloatingPointError(
            f"Matched Adam failed at step {step}: {error}"
        ) from error
