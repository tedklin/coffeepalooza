"""g-h filter on 1-state variable"""


class GHFilter:

    def __init__(self, initial_val, initial_rate, g, h):
        self._estimated_val = initial_val
        self._estimated_rate = initial_rate
        self.g = g
        self.h = h

    def set_g(self, g):
        self.g = g

    def set_h(self, h):
        self.h = h

    def update(self, measured_val, dt):
        """
        Update the filter with a new measurement. 'Prediction' also occurs in this function.
        Does not return the estimated value. In some cases, it is beneficial to update the filter at a faster
        rate than that with which you reference the value.

        :param measured_val: new measured value
        :param dt: difference in time between previous update step and current update step
        :return: nothing; use get_estimation() to get the estimated value
        """
        predicted_val = self._estimated_val + self._estimated_rate * dt
        predicted_rate = self._estimated_rate

        residual = measured_val - predicted_val
        self._estimated_rate = predicted_rate + self.h * residual / dt
        self._estimated_val = predicted_val + self.g * residual

    def get_estimation(self):
        return self._estimated_val

    def get_rate(self):
        return self._estimated_rate
