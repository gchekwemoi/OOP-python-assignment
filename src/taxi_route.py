
"""Taxi route revenue analysis for Kampala."""

import statistics as stats
import numpy as np


class Route:
    """Represent a taxi route with passenger counts and fare."""

    def __init__(self, name: str, passengers, fare: float):
        self.name = name
        self.passengers = np.asarray(passengers, dtype=float)
        self.fare = float(fare)

        if self.passengers.ndim != 1 or len(self.passengers) == 0:
            raise ValueError("Provide at least one passenger count.")

        if not np.all(np.isfinite(self.passengers)):
            raise ValueError("Passenger counts must be finite.")

        if np.any(self.passengers < 0):
            raise ValueError("Passenger counts cannot be negative.")

        if not np.all(self.passengers == np.floor(self.passengers)):
            raise ValueError("Passenger counts must be whole numbers.")

        if not np.isfinite(self.fare) or self.fare <= 0:
            raise ValueError("Fare must be a positive number.")

    def daily_revenue(self) -> np.ndarray:
        """Calculate daily revenue in UGX."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        """Calculate total revenue across all observations."""
        return float(np.sum(self.daily_revenue()))

    def summary_statistics(self) -> dict:
        """Calculate passenger demand statistics."""
        values = self.passengers.tolist()

        return {
            "mean": stats.mean(values),
            "variance": stats.variance(values) if len(values) > 1 else 0,
            "standard_deviation": (
                stats.stdev(values) if len(values) > 1 else 0
            )
        }

    def __repr__(self) -> str:
        return (
            f"Route(name={self.name!r}, "
            f"fare=UGX {self.fare:,.0f}, "
            f"days={len(self.passengers)})"
        )
