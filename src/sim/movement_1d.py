"""Implementation of a Kalman filter on 1-dimensional movement"""

import math
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from filterpy.common import discretization
from filters.kalman_filter import KalmanFilter


def simulate(filter, z, dt):
    """
    :param filter:
    :param z:
    :param dt:
    :return: numpy array of state estimates, numpy array of covariances, numpy array of times
    """
    state_estimates = []
    covariances = []
    times = []
    time = 0
    for measurement in z:
        time += dt
        filter.update(measurement, dt)
        print("time: " + str(time))
        print("state estimate: ")
        print(filter.get_state())
        print()
        state_estimates.append(filter.get_state())
        covariances.append(filter.get_state_covariance())
        times.append(time)
    return np.array(state_estimates), np.array(covariances), np.array(times)


def const_vel_data_gen(initial_pos, desired_vel, measurement_var, process_var, num_steps, dt):
    """
    Constant velocity movement with injected noise in the process (velocity) and the measurement (position)
    :param initial_pos:
    :param desired_vel:
    :param measurement_var:
    :param process_var:
    :param num_steps:
    :param dt:
    :return: numpy array of actual positions, numpy array of measured positions
    """
    pos = initial_pos
    x = []
    z = []
    for step in range(num_steps):
        vel = desired_vel + (randn() * math.sqrt(process_var))
        pos += vel * dt
        measured_pos = pos + (randn() * math.sqrt(measurement_var))
        x.append(pos)
        z.append(measured_pos)
    return np.array(x), np.array(z)


def const_accel_data_gen(initial_pos, initial_vel, desired_accel, measurement_var, process_var, num_steps, dt):
    """
    Constantly accelerated movement with injected noise in the process (acceleration) and the measurement (position)
    :param initial_pos:
    :param desired_accel:
    :param measurement_var:
    :param process_var:
    :param num_steps:
    :param dt:
    :return: array of actual positions, array of measured positions
    """
    pos = initial_pos
    vel = initial_vel
    x = []
    z = []
    for step in range(num_steps):
        accel = desired_accel + (randn() * math.sqrt(process_var))
        vel += accel * dt
        pos += vel * dt
        measured_pos = pos + (randn() * math.sqrt(measurement_var))
        x.append(pos)
        z.append(measured_pos)
    return np.array(x), np.array(z)


num_steps = 50
dt = 1.
measurement_var = 10
process_var = 0.01
x = np.array([[10.0, 4.5]]).T
P = np.diag([500, 49])
F = np.array([[1, dt],
              [0, 1]])
H = np.array([[1.0, 0.0]])
R = np.array([[measurement_var]])
Q = discretization.Q_discrete_white_noise(2, dt, process_var)
filter = KalmanFilter(x, P, F, H, R, Q)
actual_data, measured_data = const_vel_data_gen(0.0, 1.0, measurement_var, process_var, num_steps, dt)
state_estimates, covariances, times = simulate(filter, measured_data, dt)
