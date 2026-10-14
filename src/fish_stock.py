
"""Fish population and harvesting model for Lake Victoria."""

import numpy as np


class FishStock:
    """Simulate fish stock using logistic growth and harvesting."""

    def __init__(
        self,
        initial_stock: float = 4000,
        growth_rate: float = 0.4,
        carrying_capacity: float = 10000
    ):
        if not np.isfinite(initial_stock) or initial_stock < 0:
            raise ValueError("Initial stock must be non-negative.")

        if not np.isfinite(growth_rate) or growth_rate < 0:
            raise ValueError("Growth rate must be non-negative.")

        if not np.isfinite(carrying_capacity) or carrying_capacity <= 0:
            raise ValueError("Carrying capacity must be positive.")

        self.initial_stock = float(initial_stock)
        self.growth_rate = float(growth_rate)
        self.carrying_capacity = float(carrying_capacity)

    def simulate(self, harvest_rate: float, weeks: int = 52):
        """Return weekly fish stocks and harvest quantities."""

        if not np.isfinite(harvest_rate) or not 0 <= harvest_rate <= 1:
            raise ValueError("Harvest rate must be between 0 and 1.")

        if not isinstance(weeks, int) or weeks < 1:
            raise ValueError("Weeks must be a positive integer.")

        stocks = [self.initial_stock]
        harvests = []

        for _ in range(weeks):
            current = stocks[-1]

            harvest = harvest_rate * current

            growth = (
                self.growth_rate
                * current
                * (1 - current / self.carrying_capacity)
            )

            next_stock = max(0.0, current + growth - harvest)

            harvests.append(harvest)
            stocks.append(next_stock)

        return {
            "stocks": np.array(stocks),
            "harvests": np.array(harvests)
        }

    def __repr__(self):
        return (
            f"FishStock(initial_stock={self.initial_stock}, "
            f"growth_rate={self.growth_rate}, "
            f"carrying_capacity={self.carrying_capacity})"
        )
