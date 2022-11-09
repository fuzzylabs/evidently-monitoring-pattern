"""Probability distribution class."""
import numpy as np


class ProbDistribution:
    """Store and generate values based on the skewed distribution of a feature."""

    def __init__(self, dist: dict) -> None:
        """Initialise class variables.

        Args:
            dist (dict): distribution of a feature e.g. {Apple: 0.3, Banana: 0.5, Pear: 0.2}
        """
        self.no_items = list(dist.keys())
        self.items_dist = list(dist.values())
        self.shuffled_dist = self.skew_dist(self.items_dist)

    def skew_dist(self, items_dist: list) -> dict:
        """Skew the distrubtion.

        Args:
            items_dist (list): the distribution probability

        Returns:
            dict: a skew distribution e.g. {Apple: 0.0, Banana: 0.0, Pear: 1.0}
        """
        min_dist_idx = items_dist.index(min(items_dist))
        shuffled_dist = [0 for x in range(len(items_dist))]
        shuffled_dist[min_dist_idx] = 1.0

        return shuffled_dist

    def generate_val(self, shuffle_dist: bool = False) -> float:
        """Generate a value base on the probability distribution.

        Args:
            shuffle_dist (bool): whether to use skewed distribution or not

        Returns:
            float: a value based on the probability distribution used
        """
        if not shuffle_dist:
            val = np.random.choice(self.no_items, p=self.items_dist)
        else:
            val = np.random.choice(self.no_items, p=self.shuffled_dist)

        return val
