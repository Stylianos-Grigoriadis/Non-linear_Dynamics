import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import math

def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    x, y, z: Points
    s, r, b: Parameters defining the Lorenz attractor

    x_dot, y_dot, z_dot: Values of the Lorenz attractor's partial derivatives
    at the point x, y, z
    '''

    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z

    return x_dot, y_dot, z_dot


