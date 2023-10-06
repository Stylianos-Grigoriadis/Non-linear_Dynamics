import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import math
import lib
import Non_linear_lib as nll
import colorednoise as cn

# sine wave
x = np.arange(0,10000)
y = np.sin(x/5)

beta = 0    # the exponent: 0=white noite; 1=pink noise;  2=red noise (also "brownian noise")
samples = 10000  # number of samples to generate (time series extension)
#a = cn.powerlaw_psd_gaussian(beta, samples)

# White noise
rand_data = cn.powerlaw_psd_gaussian(0, 10000)

# Pink noise
pink_data = cn.powerlaw_psd_gaussian(1, 10000)

# Create a Lorenz System
dt = 0.01
num_steps = 10000

# Initial values require one or more
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Initial values setting
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = nll.lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

def ploting(y):
    periodic_first_derivative = lib.derivative(y,10000)
    dx1 = y[:-1]

    periodic_second_derivative = lib.derivative(periodic_first_derivative,10000)
    periodic_first_derivative1 = periodic_first_derivative[:-1]
    dx2 = dx1[:-1]


    figure=plt.figure()
    plt.suptitle('Periodic Signal')
    ax1 = figure.add_subplot(3, 2, 1)

    # For Sine Function
    ax1.plot(y)
    ax1.set_title("Time series")

    ax2 = figure.add_subplot(3, 2, 2)
    ax2.plot(dx1, periodic_first_derivative)
    ax2.set_title("First derivative")

    ax3 = figure.add_subplot(3, 2, (3,6), projection='3d')
    ax3.plot(dx2, periodic_first_derivative1, periodic_second_derivative)
    ax3.set_title("Second derivative")

    plt.show()

# ploting(y)
# lib.DFA(rand_data)
# ploting(rand_data)
# lib.DFA(pink_data)
# ploting(pink_data)
# ploting(zs)
