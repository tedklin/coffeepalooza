"""Recursive discrete Bayesian estimation"""

import numpy as np
from scipy.ndimage import filters
from util.caffeinated_math import normalize


class BayesFilter:

    def __init__(self, prior, known_map, ):
        self._belief = prior
        self._known_map = known_map
        self._likelihood = np.ones(len(self.known_map))

    def update(self, movement, kernel, measurement, measurement_accuracy):
        """
        Update the filter with new sensor readings.
        :param movement: prediction for relative movement from last update
        :param kernel: represents the uncertainty in prediction
        :param measurement: sensor reading for absolute positioning
        :param measurement_accuracy: the probability (0 <= measurement_prob < 1) that the measurement is correct
        :return: nothing; use get_belief() to get the belief probability distribution
        """

        # predict step
        # wrap means that our belief wraps from the highest index to the first index.
        # this follows the circular room map assumption.
        prior = filters.convolve(np.roll(self._belief, movement), kernel, mode='wrap')

        # compute scale
        if measurement_accuracy >= 1:
            measurement_accuracy = 0.9999
        elif measurement_accuracy < 0:
            measurement_accuracy = 0
        scale = measurement_accuracy / (1 - measurement_accuracy)

        # update step
        self._likelihood[self._known_map == measurement] *= scale
        posterior = self._likelihood * prior
        self._belief = normalize(posterior)

    def get_belief(self):
        return self._belief