import numpy as np


class DistrictPopulation:
    """Store a district's yearly populations, measured in thousands."""

    def __init__(
        self,
        name: str,
        years: list[int],
        populations: list[float]
    ) -> None:
        self.name = name
        self.years = np.array(years, dtype=int)
        self.populations = np.array(populations, dtype=float)

        if self.years.ndim != 1 or self.populations.ndim != 1:
            raise ValueError("Years and populations must be one-dimensional.")

        if len(self.years) != len(self.populations):
            raise ValueError("Years and populations must have equal lengths.")

        if len(self.years) == 0:
            raise ValueError("Data must not be empty.")

        if not np.all(np.isfinite(self.populations)):
            raise ValueError("Populations must contain only finite numbers.")

        if np.any(self.populations < 0):
            raise ValueError("Populations cannot be negative.")

        if np.any(np.diff(self.years) != 1):
            raise ValueError("Years must be consecutive and increasing.")

    def __len__(self) -> int:
        """Return the number of yearly observations."""
        return len(self.years)

    def __repr__(self) -> str:
        """Return a short description of the district."""
        return f"DistrictPopulation(name={self.name!r}, observations={len(self)})"
