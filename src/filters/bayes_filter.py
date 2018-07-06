"""Recursive discrete Bayesian estimation"""

import numpy as np
from scipy.ndimage import filters
from util.caffeinated_math import normalize


class BayesFilter:

    def __init__(self, prior, known_map, ):
        self._belief = prior
        self._known_map = known_map
        self._likelihood = np.ones(len(self.known_map))

    def update(self, movement, kernel, landmark_detected, landmark_probability):
        """
        Update the filter with new sensor readings.
        :param movement: sensor reading for relative movement from last update
        :param kernel: represents the noise in the relative movement sensor
        :param landmark_detected: sensor reading for absolute location using landmarks
        :param landmark_probability: the probability (0 <= measurement_prob < 1) that the landmark detection is correct
        :return:
        """

        # predict step
        # wrap means that our belief wraps from the highest index to the first index.
        # this follows the circular room map assumption.
        prior = filters.convolve(np.roll(self._belief, movement), kernel, mode='wrap')

        # compute likelihood
        if landmark_probability >= 1:
            landmark_probability = 0.9999
        elif landmark_probability < 0:
            landmark_probability = 0
        scale = landmark_probability / (1 - landmark_probability)
        self._likelihood[self._known_map == landmark_detected] *= scale

        # update step
        posterior = normalize(self._likelihood * prior)
        self._belief = posterior

    def get_belief(self):
        return self._belief