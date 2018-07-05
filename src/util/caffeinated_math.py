"""Useful math functions"""

import numpy as np


def normalize(distribution):
    """
    Divide each element in an array by the sum of all of its elements.
    Creates a probability distribution that sums to 1 while maintaining the ratios among elements.

    :param distribution: numpy array
    :return: normalized distribution
    """
    distribution /= sum(np.asarray(distribution, dtype=float))
    return distribution
