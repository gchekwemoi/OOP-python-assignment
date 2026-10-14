
"""Tests for the Lake Victoria Fish Stock and Export Risk Model."""

import numpy as np
import pytest

from src.fish_stock import FishStock
from src.price_model import PriceModel
from src.risk_assessor import RiskAssessor


def test_fish_stock_simulation():
    """Check fish stock and harvest output."""
    model = FishStock()
    result = model.simulate(harvest_rate=0.10, weeks=52)

    assert len(result["stocks"]) == 53
    assert len(result["harvests"]) == 52
    assert np.all(result["stocks"] >= 0)


def test_fish_growth_calculation():
    """Check the logistic growth equation by hand."""
    model = FishStock(
        initial_stock=4000,
        growth_rate=0.4,
        carrying_capacity=10000
    )

    result = model.simulate(harvest_rate=0.10, weeks=1)

    # Growth = 0.4 * 4000 * (1 - 4000/10000) = 960
    # Harvest = 0.10 * 4000 = 400
    # Next stock = 4000 + 960 - 400 = 4560
    assert result["stocks"][1] == pytest.approx(4560)
    assert result["harvests"][0] == pytest.approx(400)


def test_invalid_harvest_rate():
    """Reject a harvesting rate above 100%."""
    model = FishStock()

    with pytest.raises(ValueError):
        model.simulate(harvest_rate=1.5)


def test_price_reproducibility():
    """The same random seed should generate the same prices."""
    model = PriceModel(seed=42)

    first = model.simulate(weeks=52)
    second = model.simulate(weeks=52)

    np.testing.assert_array_equal(first, second)


def test_price_limits():
    """Simulated fish prices must remain within the limits."""
    model = PriceModel()
    prices = model.simulate(weeks=52)

    assert np.all(prices >= 9000)
    assert np.all(prices <= 16000)


def test_revenue_calculation():
    """Check conversion from tonnes to kilograms."""
    model = PriceModel()

    revenue = model.calculate_revenue(
        harvest_tonnes=np.array([100]),
        prices=np.array([12000])
    )

    assert revenue[0] == pytest.approx(1_200_000_000)


def test_risk_classification():
    """Check the three illustrative CV risk categories."""
    assessor = RiskAssessor()

    assert assessor.classify(5) == "Low"
    assert assessor.classify(15) == "Moderate"
    assert assessor.classify(30) == "High"


def test_revenue_statistics():
    """Check revenue mean and standard deviation."""
    assessor = RiskAssessor()

    result = assessor.summarise([100, 200, 300])

    assert result["mean"] == pytest.approx(200)
    assert result["median"] == pytest.approx(200)
    assert result["standard_deviation"] == pytest.approx(100)


def test_value_at_risk():
    """Check the 5th percentile of annual revenue."""
    assessor = RiskAssessor()

    result = assessor.value_at_risk(
        np.array([100, 200, 300, 400, 500])
    )

    assert result["revenue_5th_percentile"] == pytest.approx(120)
    assert result["mean_annual_revenue"] == pytest.approx(300)
    assert result["shortfall_from_mean"] == pytest.approx(180)


def test_empty_revenue():
    """Reject an empty revenue dataset."""
    assessor = RiskAssessor()

    with pytest.raises(ValueError):
        assessor.summarise([])
