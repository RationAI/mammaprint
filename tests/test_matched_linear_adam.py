from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from ml.matched_linear_adam import fit_matched_adam
from scripts.ml.compare_ridge_historical_adam import fit_adam as fit_historical_adam


def objective(x, y, alpha, weights, intercept):
    residual = x @ weights + intercept - y
    return (residual @ residual + alpha * (weights @ weights)) / len(y)


def test_updates_match_historical_torch_adam_on_the_same_objective():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(11, 19)) + np.arange(19)
    y = rng.normal(size=11) + 1.0
    alpha, steps, rate = 0.13, 61, 0.014
    ridge = Ridge(alpha=alpha, solver="svd").fit(x, y)
    optimum = objective(x, y, alpha, ridge.coef_, ridge.intercept_)

    fit = fit_matched_adam(x, y, alpha, optimum, max_steps=steps, lr=rate)
    expected_weights, expected_intercept = fit_historical_adam(x, y, alpha, steps, rate)

    assert fit.steps == steps
    assert not fit.converged
    np.testing.assert_allclose(fit.weights, expected_weights, rtol=0, atol=1e-12)
    assert fit.intercept == pytest.approx(expected_intercept, abs=1e-12)
    assert fit.objective == pytest.approx(
        objective(x, y, alpha, fit.weights, fit.intercept)
    )


def test_matches_the_exact_ridge_optimum_with_an_unpenalized_intercept():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(75, 4)) + np.array([3.0, -2.0, 5.0, -10.0])
    y = x @ np.array([0.03, -0.02, 0.01, 0.04]) + 0.5 + rng.normal(0, 0.02, 75)
    alpha, tolerance = 0.4, 1e-8
    ridge = Ridge(alpha=alpha, solver="svd").fit(x, y)
    optimum = objective(x, y, alpha, ridge.coef_, ridge.intercept_)

    fit = fit_matched_adam(x, y, alpha, optimum, relative_tolerance=tolerance)

    assert fit.converged
    assert 0 < fit.steps < 2000 and fit.steps % 25 == 0
    assert fit.objective - optimum <= abs(optimum) * tolerance
    np.testing.assert_allclose(fit.weights, ridge.coef_, atol=3e-6)
    np.testing.assert_allclose(
        x @ fit.weights + fit.intercept, ridge.predict(x), atol=1e-5
    )


def test_constant_targets_are_optimal_without_an_update():
    x = np.arange(12.0).reshape(4, 3)
    y = np.full(4, 2.0)
    fit = fit_matched_adam(x, y, 0.1, 0.0)

    assert fit.steps == 0 and fit.converged
    np.testing.assert_array_equal(fit.weights, np.zeros(3))
    assert fit.intercept == 2.0
    assert fit.objective == 0.0


def test_rejects_nonfinite_training_data_or_invalid_optimum():
    with pytest.raises(ValueError, match="finite training features and targets"):
        fit_matched_adam(np.array([[np.inf], [0.0]]), np.array([0.5, 0.4]), 0.1, 0.0)

    with pytest.raises(ValueError, match="below the supplied optimum"):
        fit_matched_adam(np.array([[1.0], [1.0]]), np.array([0.0, 1.0]), 0.1, 10.0)


def test_reports_when_finite_data_produces_a_nonfinite_objective():
    with pytest.raises(FloatingPointError, match="step 0: nonfinite initial objective"):
        fit_matched_adam(np.array([[0.5], [0.5]]), np.array([1e250, -1e250]), 0.1, 0.0)
