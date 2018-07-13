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


class Gaussian:
    """Gaussian represented by mean and variance. Gaussians can be added or multiplied to form a new Gaussian."""

    def __init__(self, mean, variance):
        self._mean = mean
        self._variance = variance

    def set_mean(self, mean):
        self._mean = mean

    def set_variance(self, variance):
        self._variance = variance

    def get_mean(self):
        return self._mean

    def get_variance(self):
        return self._variance

    def add_by_gaussian(self, gaussian_addend):
        self._mean += gaussian_addend.get_mean()
        self._variance += gaussian_addend.get_variance()

    def multiply_by_gaussian(self, gaussian_multiplier):
        self._mean = (self._variance * gaussian_multiplier.get_mean() + gaussian_multiplier.get_variance() *
                      self._mean) / (self._variance + gaussian_multiplier.get_variance())
        self._variance = (self._variance * gaussian_multiplier.get_variance()) / (self._variance +
                                                                                  gaussian_multiplier.get_variance())

    def multiply_by_scalar(self, scalar):
        self._mean *= scalar
        self._variance *= scalar
