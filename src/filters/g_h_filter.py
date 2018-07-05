import numpy as np

'''
g-h filter on 1-state variable
'''


class GHFilter:

    def __init__(self, initial_val, initial_rate, g, h):
        self.estimated_val = initial_val
        self.estimated_rate = initial_rate
        self.g = g
        self.h = h

    def set_g(self, g):
        self.g = g

    def set_h(self, h):
        self.h = h

    def update(self, measured_val, dt):
        predicted_val = self.estimated_val + self.estimated_rate * dt
        predicted_rate = self.estimated_rate

        residual = measured_val - predicted_val
        self.estimated_rate = predicted_rate + self.h * residual / dt
        self.estimated_val = predicted_val + self.g * residual

    def get_estimation(self):
        return self.estimated_val

    def get_rate(self):
        return self.estimated_rate
