"""Multivariate Kalman filter."""

import numpy as np
import scipy


class KalmanFilter:

    def __init__(self, x, P, F, H, R, Q):
        """
        :param x: state vector
        :param P: state covariance matrix
        :param F: state transition function
        :param H: measurement function
        :param R: measurement noise covariance matrix
        :param Q: process noise covariance matrix
        """
        self._x = x
        self._P = P
        self._F = F
        self._H = H
        self._R = R
        self._Q = Q

    def update(self, z, dt):
        """
        :param z: measurement
        :param dt: time step
        :return:
        """
        # predict
        self._x = np.dot(self._F, self._x)
        self._P = np.dot(self._F, self._P).dot(self._F.T) + self._Q

        # update
        S = np.dot(self._H, self._P).dot(self._H.T) + self._R
        K = np.dot(self._P, self._H.T).dot(scipy.linalg.inv(S))
        y = z - np.dot(self._H, self._x)
        self._x += np.dot(K, y)
        self._P = self._P - np.dot(K, self._H).dot(self._P)

    def get_state(self):
        """
        :return: state vector
        """
        return self._x

    def get_state_covariance(self):
        """
        :return: state covariance matrix
        """
        return self._P
