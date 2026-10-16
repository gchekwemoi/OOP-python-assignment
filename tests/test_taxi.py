
"""Tests for the Taxi Route Revenue and Fleet Planner."""

import numpy as np
import pytest

from src.taxi_route import Route
from src.taxi_forecasting import (
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    LinearTrendForecaster
)


def test_route_daily_revenue():
    route = Route("Ntinda", [35, 40, 42], 2000)

    np.testing.assert_array_equal(
        route.daily_revenue(),
        [70000, 80000, 84000]
    )


def test_route_total_revenue():
    route = Route("Ntinda", [35, 40, 42], 2000)

    assert route.total_revenue() == pytest.approx(234000)


def test_route_statistics():
    route = Route("Test", [35, 40, 45], 2000)

    stats = route.summary_statistics()

    assert stats["mean"] == pytest.approx(40)
    assert stats["variance"] == pytest.approx(25)
    assert stats["standard_deviation"] == pytest.approx(5)


def test_negative_passengers():
    with pytest.raises(ValueError):
        Route("Ntinda", [35, -5, 42], 2000)


def test_invalid_fare():
    with pytest.raises(ValueError):
        Route("Ntinda", [35, 40, 42], 0)


def test_moving_average():
    model = MovingAverageForecaster()

    model.fit([52, 47, 45])

    assert model.predict() == pytest.approx(48)


def test_exponential_smoothing():
    model = ExponentialSmoothingForecaster(alpha=0.5)

    model.fit([10, 20, 30])

    # Level: 10 -> 15 -> 22.5
    assert model.predict() == pytest.approx(22.5)


def test_linear_trend():
    model = LinearTrendForecaster()

    model.fit([10, 20, 30, 40])

    assert model.predict() == pytest.approx(50)


def test_invalid_alpha():
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(alpha=1.5)


def test_predict_before_fit():
    model = MovingAverageForecaster()

    with pytest.raises(ValueError):
        model.predict()
