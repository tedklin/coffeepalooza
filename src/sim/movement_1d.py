"""Implementation of a Kalman filter on 1-dimensional movement"""

import math
import numpy as np
from numpy.random import randn


def const_vel_data_gen(initial_pos, desired_vel, measurement_var, process_var, num_steps, dt):
    """
    Data for constant velocity movement with injected noise in the process (velocity) and the measurement (position)
    :param initial_pos:
    :param desired_vel:
    :param measurement_var:
    :param process_var:
    :param num_steps:
    :param dt:
    :return: array of actual positions, array of measured positions
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
    Data for constantly accelerated movement with injected noise in the process (acceleration) and the measurement (position)
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
