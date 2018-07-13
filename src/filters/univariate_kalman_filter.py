"""Univariate (1-dimensional) Kalman filter on a single state."""

import time
from util.caffeinated_math import Gaussian


class BayesianUnivariateKalmanFilter:
    """Derived from Bayesian principles"""

    def __init__(self, initial_belief):
        """
        :param initial_belief: represented by a Gaussian object
        """
        self._belief = initial_belief
        self._time = time.perf_counter()
        self._movement = Gaussian(0, 0)  # don't want to create a new Gaussian object every iteration
        self._likelihood = Gaussian(0, 0)

    def update(self, velocity, process_variance, measurement, sensor_variance):
        """
        Update the Kalman filter with predicted movement and sensor measurement
        :param velocity:
        :param process_variance:
        :param measurement:
        :param sensor_variance:
        :return:
        """
        # predict step using a really simple model (dx = vel * dt)
        dt = time.perf_counter() - self._time
        self._movement.set_mean(velocity * dt)
        self._movement.set_variance(process_variance * dt)
        self._belief.add_by_gaussian(self._movement)

        # update step
        self._likelihood.set_mean(measurement)
        self._likelihood.set_variance(sensor_variance)
        self._belief.multiply_by_gaussian(self._likelihood)

    def get_belief(self):
        return self._belief

    def get_estimate(self):
        return self._belief.get_mean()
