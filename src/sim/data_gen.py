import math
import numpy as np
from numpy.random import randn

def gen_const_vel_data(initial_pos, desired_vel, measurement_var, process_var, num_steps, dt):
    """
    Data for constant velocity movement with injected noise in the process (velocity) and the measurement (position)
    :param initial_pos:
    :param vel:
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