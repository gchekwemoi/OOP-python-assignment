
"""Passenger demand forecasting for Kampala taxi routes."""

from abc import ABC, abstractmethod

import numpy as np


class TaxiForecaster(ABC):
    """Abstract base class for passenger forecasting models."""

    @abstractmethod
    def fit(self, passengers):
        """Learn from historical passenger counts."""
        pass

    @abstractmethod
    def predict(self) -> float:
        """Predict passenger demand for the next day."""
        pass

    @staticmethod
    def validate_data(passengers):
        """Validate passenger observations."""
        values = np.asarray(passengers, dtype=float)

        if values.ndim != 1 or len(values) < 3:
            raise ValueError(
                "Provide at least three passenger observations."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Passenger counts must be finite."
            )

        if np.any(values < 0):
            raise ValueError(
                "Passenger counts cannot be negative."
            )

        return values


class MovingAverageForecaster(TaxiForecaster):
    """Forecast using the average of the last three days."""

    def fit(self, passengers):
        self.values = self.validate_data(passengers)

    def predict(self) -> float:
        if not hasattr(self, "values"):
            raise ValueError("Fit the model before predicting.")

        return float(np.mean(self.values[-3:]))


class ExponentialSmoothingForecaster(TaxiForecaster):
    """Forecast passenger demand using exponential smoothing."""

    def __init__(self, alpha: float = 0.5):
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1.")

        self.alpha = float(alpha)

    def fit(self, passengers):
        self.values = self.validate_data(passengers)

        # Initialise the smoothed value
        level = float(self.values[0])

        # Update using each subsequent observation
        for value in self.values[1:]:
            level = (
                self.alpha * value
                + (1 - self.alpha) * level
            )

        self.level = level

    def predict(self) -> float:
        if not hasattr(self, "level"):
            raise ValueError("Fit the model before predicting.")

        return float(self.level)


class LinearTrendForecaster(TaxiForecaster):
    """Forecast passenger demand using a linear trend."""

    def fit(self, passengers):
        self.values = self.validate_data(passengers)

        days = np.arange(1, len(self.values) + 1)

        self.slope, self.intercept = np.polyfit(
            days,
            self.values,
            1
        )

        self.next_day = len(self.values) + 1

    def predict(self) -> float:
        if not hasattr(self, "next_day"):
            raise ValueError("Fit the model before predicting.")

        forecast = (
            self.slope * self.next_day
            + self.intercept
        )

        return float(max(0, forecast))
