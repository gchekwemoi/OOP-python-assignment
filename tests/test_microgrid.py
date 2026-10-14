
"""Tests for the Solar Micro-Grid Dispatch Planner."""

import pytest

from src.microgrid import MicroGrid


def test_valid_allocation():
    """Check a feasible solar and battery allocation."""
    grid = MicroGrid()
    result = grid.solve_day(80, 90)

    assert result["solar_kwh"] == pytest.approx(20)
    assert result["battery_kwh"] == pytest.approx(10)
    assert result["feasible"] is True


def test_zero_demand():
    """Zero demand should require zero energy."""
    grid = MicroGrid()
    result = grid.solve_day(0, 0)

    assert result["solar_kwh"] == pytest.approx(0)
    assert result["battery_kwh"] == pytest.approx(0)
    assert result["feasible"] is True


def test_negative_demand():
    """Negative demand should be rejected."""
    grid = MicroGrid()

    with pytest.raises(ValueError):
        grid.solve_day(-10, 50)


def test_non_numeric_demand():
    """Text input should be rejected."""
    grid = MicroGrid()

    with pytest.raises(ValueError):
        grid.solve_day("invalid", 50)


def test_infeasible_allocation():
    """Flag an allocation requiring negative battery energy."""
    grid = MicroGrid()
    result = grid.solve_day(40, 90)

    assert result["feasible"] is False
    assert result["battery_kwh"] < 0


def test_singular_matrix():
    """Reject a system without a unique solution."""
    with pytest.raises(ValueError):
        MicroGrid([[2, 4], [1, 2]])
