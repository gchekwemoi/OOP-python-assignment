
"""Tests for the Rainfall and Crop Suitability Analyser."""

import numpy as np
import pytest

from src.rainfall import Region
from src.crop_rule import CropRule
from src.rainfall_analysis import RainfallAnalyser


def test_region_annual_total():
    region = Region("Test", [100] * 12)
    assert region.annual_total() == pytest.approx(1200)


def test_region_monthly_mean():
    region = Region("Test", [100] * 12)
    assert region.monthly_mean() == pytest.approx(100)


def test_invalid_rainfall():
    with pytest.raises(ValueError):
        Region("Test", [100, 200])


def test_negative_rainfall():
    with pytest.raises(ValueError):
        Region("Test", [100] * 11 + [-5])


def test_crop_suitability():
    maize = CropRule("Maize", 100, 200)

    assert maize.classify(150) == "Suitable rainfall"
    assert maize.classify(50) == "Drought risk"
    assert maize.classify(250) == "Excess rainfall risk"


def test_invalid_crop_thresholds():
    with pytest.raises(ValueError):
        CropRule("Maize", 200, 100)


def test_cosine_similarity():
    a = np.array([1, 2, 3])
    b = np.array([2, 4, 6])

    result = RainfallAnalyser.cosine_similarity(a, b)

    assert result == pytest.approx(1.0)


def test_cosine_validation():
    a = np.array([100, 150, 200])
    b = np.array([120, 170, 220])

    assert RainfallAnalyser.validate_cosine(a, b)


def test_euclidean_distance():
    a = np.array([1, 2, 3])
    b = np.array([4, 6, 3])

    result = RainfallAnalyser.euclidean_distance(a, b)

    assert result == pytest.approx(5.0)


def test_rainy_season_detection():
    rainfall = [
        120, 140, 180, 200, 220, 180,
        90, 70, 60, 100, 110, 130
    ]

    region = Region("Kampala", rainfall)

    result = RainfallAnalyser.detect_rainy_seasons(region)

    assert "May" in result["peak_months"]
    assert result["pattern"] == "Bimodal"
