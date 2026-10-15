
"""Rainfall similarity and rainy-season analysis."""

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine


class RainfallAnalyser:
    """Compare regional rainfall patterns and detect rainy seasons."""

    def __init__(self, regions: list):
        """Accept a collection of Region objects."""

        if len(regions) < 2:
            raise ValueError("Provide at least two regions.")

        self.regions = regions

    @staticmethod
    def cosine_similarity(a, b) -> float:
        """Calculate vector cosine similarity using NumPy."""

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            raise ValueError("Cosine similarity requires non-zero vectors.")

        return float(
            np.dot(a, b)
            / (np.linalg.norm(a) * np.linalg.norm(b))
        )

    @staticmethod
    def pearson_correlation(a, b) -> float:
        """Calculate the Pearson correlation coefficient."""

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if np.std(a) == 0 or np.std(b) == 0:
            raise ValueError("Correlation requires varying observations.")

        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def euclidean_distance(a, b) -> float:
        """Calculate the Euclidean distance between rainfall vectors."""

        return float(np.linalg.norm(
            np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        ))

    def similarity_matrices(self) -> dict:
        """Compare each region with every other region."""

        names = [region.name for region in self.regions]
        rainfall = [region.rainfall for region in self.regions]
        n = len(self.regions)

        matrices = {
            "Cosine similarity": np.zeros((n, n)),
            "Pearson correlation": np.zeros((n, n)),
            "Euclidean distance": np.zeros((n, n))
        }

        for i in range(n):
            for j in range(n):
                matrices["Cosine similarity"][i, j] = (
                    self.cosine_similarity(rainfall[i], rainfall[j])
                )
                matrices["Pearson correlation"][i, j] = (
                    self.pearson_correlation(rainfall[i], rainfall[j])
                )
                matrices["Euclidean distance"][i, j] = (
                    self.euclidean_distance(rainfall[i], rainfall[j])
                )

        return {
            "region_names": names,
            "matrices": matrices
        }

    @staticmethod
    def validate_cosine(a, b) -> bool:
        """Verify our cosine formula against SciPy."""

        calculated = RainfallAnalyser.cosine_similarity(a, b)
        scipy_result = 1 - cosine(a, b)

        return bool(np.isclose(calculated, scipy_result))

    @staticmethod
    def detect_rainy_seasons(region) -> dict:
        """Identify rainfall peaks and classify their pattern."""

        rainfall = region.rainfall

        # Treat Jan-Dec as a repeating yearly cycle.
        # Repeating the array allows peaks near December/January.
        extended = np.concatenate([rainfall, rainfall, rainfall])

        peaks, _ = find_peaks(
            extended,
            prominence=25,
            distance=2
        )

        # Keep peaks in the middle year only.
        middle_peaks = peaks[
            (peaks >= 12) & (peaks < 24)
        ] - 12

        peak_months = [
            region.MONTHS[index]
            for index in middle_peaks
        ]

        if len(peak_months) >= 2:
            pattern = "Bimodal"
        elif len(peak_months) == 1:
            pattern = "Unimodal"
        else:
            pattern = "No clear peak"

        return {
            "peak_months": peak_months,
            "pattern": pattern
        }
