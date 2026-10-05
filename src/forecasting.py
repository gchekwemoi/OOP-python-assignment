from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """Define the methods every forecasting model must provide."""

    @abstractmethod
    def fit(self, years: np.ndarray, populations: np.ndarray) -> None:
        """Learn from historical population data."""
        pass

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the requested number of future years."""
        pass

    def validate_data(self, years, populations):
        """Check that training data is suitable for forecasting."""
        years = np.asarray(years, dtype=float)
        populations = np.asarray(populations, dtype=float)

        if years.ndim != 1 or populations.ndim != 1:
            raise ValueError("Training data must be one-dimensional.")

        if len(years) != len(populations) or len(years) < 2:
            raise ValueError("Provide at least two matching observations.")

        if not np.all(np.isfinite(years)):
            raise ValueError("Years must be finite.")

        if not np.all(np.isfinite(populations)):
            raise ValueError("Populations must be finite.")

        if np.any(populations <= 0):
            raise ValueError("Training populations must be positive.")

        if np.any(np.diff(years) != 1):
            raise ValueError("Years must be consecutive and increasing.")

        return years, populations

    def validate_prediction(self, horizon: int) -> None:
        """Reject invalid horizons and prediction before fitting."""
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise ValueError("Horizon must be a positive integer.")

        if horizon <= 0:
            raise ValueError("Horizon must be a positive integer.")

        if not hasattr(self, "last_year"):
            raise ValueError("Fit the model before predicting.")


class LinearForecaster(Forecaster):
    """Forecast a constant numerical increase each year."""

    def fit(self, years: np.ndarray, populations: np.ndarray) -> None:
        years, populations = self.validate_data(years, populations)
        self.first_year = years[0]
        self.last_year = years[-1]

        # Use elapsed years to keep the calculation simple.
        self.slope, self.intercept = np.polyfit(
            years - self.first_year, populations, 1
        )

    def predict(self, horizon: int) -> np.ndarray:
        self.validate_prediction(horizon)
        future_years = self.last_year + np.arange(1, horizon + 1)
        return (
            self.slope * (future_years - self.first_year)
            + self.intercept
        )


class CAGRForecaster(Forecaster):
    """Forecast a constant percentage increase each year."""

    def fit(self, years: np.ndarray, populations: np.ndarray) -> None:
        years, populations = self.validate_data(years, populations)
        self.last_year = years[-1]
        self.last_population = populations[-1]

        intervals = years[-1] - years[0]
        self.growth_factor = (
            populations[-1] / populations[0]
        ) ** (1 / intervals)

    def predict(self, horizon: int) -> np.ndarray:
        self.validate_prediction(horizon)
        steps = np.arange(1, horizon + 1)
        return self.last_population * self.growth_factor ** steps


class FibonacciForecaster(Forecaster):
    """Apply successive Fibonacci ratios as an illustrative model."""

    def fit(self, years: np.ndarray, populations: np.ndarray) -> None:
        years, populations = self.validate_data(years, populations)
        self.last_year = years[-1]
        self.last_population = populations[-1]

    def predict(self, horizon: int) -> np.ndarray:
        self.validate_prediction(horizon)

        # Start with 1, 2: ratios are 2/1, 3/2, 5/3, ...
        previous, current = 1, 2
        population = self.last_population
        forecasts = []

        for _ in range(horizon):
            population *= current / previous
            forecasts.append(population)
            previous, current = current, previous + current

        return np.array(forecasts)
