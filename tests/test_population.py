import numpy as np
import pytest

from src.population import DistrictPopulation
from src.forecasting import LinearForecaster, CAGRForecaster


def test_valid_district():
    """Check data storage and the observation count."""
    district = DistrictPopulation(
        "Example", [2020, 2021, 2022], [100, 110, 121]
    )

    assert len(district) == 3
    assert district.name == "Example"
    np.testing.assert_array_equal(
        district.populations, [100, 110, 121]
    )


def test_negative_population():
    """Negative population must be rejected."""
    with pytest.raises(ValueError):
        DistrictPopulation("Example", [2020, 2021], [100, -10])


def test_mismatched_lengths():
    """Each year must have a population value."""
    with pytest.raises(ValueError):
        DistrictPopulation("Example", [2020, 2021], [100])


def test_empty_data():
    """Empty input must be rejected."""
    with pytest.raises(ValueError):
        DistrictPopulation("Example", [], [])


def test_cagr_forecast():
    """A 10% annual increase should continue to 133.1 and 146.41."""
    model = CAGRForecaster()
    model.fit(
        np.array([2020, 2021, 2022]),
        np.array([100, 110, 121])
    )

    np.testing.assert_allclose(
        model.predict(2), [133.1, 146.41]
    )


def test_linear_forecast():
    """An increase of 10 each year should continue to 130 and 140."""
    model = LinearForecaster()
    model.fit(
        np.array([2020, 2021, 2022]),
        np.array([100, 110, 120])
    )

    np.testing.assert_allclose(model.predict(2), [130, 140])
