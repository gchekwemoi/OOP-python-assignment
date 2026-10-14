
"""Fish price simulation for Lake Victoria."""

import numpy as np


class PriceModel:
    """Simulate weekly fish prices using a bounded random walk."""

    def __init__(
        self,
        starting_price: float = 12000,
        minimum_price: float = 9000,
        maximum_price: float = 16000,
        price_volatility: float = 500,
        seed: int = 42
    ):
        """Set up the fish price simulation."""

        values = [
            starting_price,
            minimum_price,
            maximum_price,
            price_volatility
        ]

        if not all(np.isfinite(value) for value in values):
            raise ValueError("Price values must be finite.")

        if minimum_price <= 0:
            raise ValueError("Minimum price must be positive.")

        if minimum_price >= maximum_price:
            raise ValueError("Minimum must be below maximum.")

        if not minimum_price <= starting_price <= maximum_price:
            raise ValueError("Starting price must be within the limits.")

        if price_volatility < 0:
            raise ValueError("Volatility cannot be negative.")

        self.starting_price = float(starting_price)
        self.minimum_price = float(minimum_price)
        self.maximum_price = float(maximum_price)
        self.price_volatility = float(price_volatility)
        self.seed = seed

    def simulate(self, weeks: int = 52) -> np.ndarray:
        """Generate reproducible weekly prices in UGX/kg."""

        if not isinstance(weeks, int) or isinstance(weeks, bool) or weeks < 1:
            raise ValueError("Weeks must be a positive integer.")

        rng = np.random.default_rng(self.seed)

        prices = []
        current_price = self.starting_price

        for _ in range(weeks):
            change = rng.normal(
                loc=0,
                scale=self.price_volatility
            )

            current_price = np.clip(
                current_price + change,
                self.minimum_price,
                self.maximum_price
            )

            prices.append(float(current_price))

        return np.array(prices)

    def calculate_revenue(
        self,
        harvest_tonnes: np.ndarray,
        prices: np.ndarray
    ) -> np.ndarray:
        """Calculate weekly revenue in UGX."""

        harvest_tonnes = np.asarray(harvest_tonnes, dtype=float)
        prices = np.asarray(prices, dtype=float)

        if harvest_tonnes.shape != prices.shape:
            raise ValueError(
                "Harvest and price arrays must have matching shapes."
            )

        if not (
            np.all(np.isfinite(harvest_tonnes))
            and np.all(np.isfinite(prices))
        ):
            raise ValueError("Harvest and prices must be finite.")

        if np.any(harvest_tonnes < 0) or np.any(prices < 0):
            raise ValueError(
                "Harvest quantities and prices cannot be negative."
            )

        # Convert tonnes to kilograms before calculating revenue
        return harvest_tonnes * 1000 * prices

    def __repr__(self) -> str:
        """Return a readable description of the price model."""

        return (
            f"PriceModel(starting_price={self.starting_price}, "
            f"volatility={self.price_volatility})"
        )
