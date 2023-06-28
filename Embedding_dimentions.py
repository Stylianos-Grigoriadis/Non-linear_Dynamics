import numpy as np
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def AMI_Stergiou(data, L, to_matlab=False, n_bins=0):
    """
    inputs    - data, column oriented time series
              - L, maximal lag to which AMI will be calculated
              - bins, number of bins to use in the calculation, if empty an
                adaptive formula will be used
              - to_matlab, an option for MATLAB users of the code, if MATLAB
                datatypes are needed for output, use this to have them
                returned with proper types. Default is false.

                Only use if you have 'matlab.engine' installed in your current
                Python env.

                Note: this cannot be installed through the usual conda or pip
                commands, search online to view resources to help in installing
                'matlab.engine' for Python.

    outputs   - tau, first minimum in the AMI vs lag plot
              - v_AMI, vector of AMI values and associated lags

    inputs    - x, single column array with the same length as y.
              - y, single column array with the same length as x.
    outputs   - ami, the average mutual information between the two arrays

    Remarks
    - This code uses average mutual information to find an appropriate lag
      with which to perform phase space reconstruction. It is based on a
      histogram method of calculating AMI.
    - In the case a value of tau could not be found before L the code will
      automatically re-execute with a higher value of L, and will continue to
      re-execute up to a ceiling value of L.
      """


    eps = np.finfo(float).eps  # smallest floating point value

    if isinstance(L, int):
        N = len(data)

        data = np.array(data)

        if n_bins == 0:
            bins = np.ceil((np.max(data) - np.min(data)) / (3.49 * np.nanstd(data * N ** (-1 / 3), axis=0)))
        else:
            bins = n_bins

        bins = int(bins)

        data = data - min(data)  # make all data points positive
        y = np.floor(data / (np.max(data) / (bins - eps)))
        y = np.array(y,
                     dtype=int)  # converts the vector of double vals from data2 into a list of integers from 0 to overlap (where overlap is N-L).

        v = np.zeros((L, 1))  # preallocate the vector
        overlap = N - L
        increment = 1 / overlap

        pA = sp.csr_matrix((np.full(overlap, increment), (y[0:overlap], np.ones(overlap, dtype=int)))).toarray()[:, 1]

        v = np.zeros((2, L))

        for lag in range(L):  # used to be from 0:L-1 (BS)
            v[0, lag] = lag

            pB = sp.csr_matrix(
                (np.full(overlap, increment), (y[lag:overlap + lag], np.ones(overlap, dtype=int)))).toarray()[:, 1]
            # find joint probability p(A,B)=p(x(t),x(t+time_lag))
            pAB = sp.csr_matrix((np.full(overlap, increment), (y[0:overlap], y[lag:overlap + lag])))

            (A, B) = np.nonzero(pAB)
            AB = pAB.data

            v[1, lag] = np.sum(
                np.multiply(AB, np.log2(np.divide(AB, np.multiply(pA[A], pB[B])))))  # Average Mutual Information

        tau = np.array(np.full((L, 2), -1, dtype=float))

        j = 0
        for i in range(v.shape[1] - 1):  # Finds first minimum
            if v[1, i - 1] >= v[1, i] and v[1, i] <= v[1, i + 1]:
                ami = v[1, i]
                tau[j, :] = np.array([i, ami])
                j += 1

        tau = tau[:j]  # only include filled in data.

        initial_AMI = v[1, 0]
        for i in range(v.shape[1]):  # Finds first AMI value that is 20% initial AMI
            if v[1, i] < (0.2 * initial_AMI):
                tau[0, 1] = i
                break

        v_AMI = v

        return (tau, v_AMI)
    elif isinstance(L, np.ndarray) or isinstance(L, list):
        x = data if isinstance(data, np.ndarray) else np.array(data)
        y = L if isinstance(L, np.ndarray) else np.array(L)

        if len(x) != len(y):
            raise ValueError('X and Y must be the same size.')

        increment = 1 / len(y)
        one = np.ones(len(y), dtype=int)

        bins1 = np.ceil((max(x) - min(x)) / (3.49 * np.nanstd(x) * len(x) ** (-1 / 3)))  # Scott 1979
        bins2 = np.ceil((max(y) - min(y)) / (3.49 * np.nanstd(y) * len(y) ** (-1 / 3)))  # Scott 1979
        x = x - min(x)  # make all data points positive
        x = np.floor(x / (max(x) / (bins1 - eps)))  # scaling the data
        y = y - min(y)  # make all data points positive
        y = np.floor(y / (max(y) / (bins2 - eps)))  # scaling the data

        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)

        increment = np.full(len(y), increment)
        pA = sp.csr_matrix((increment, (x, one))).toarray()[:, 1]
        pB = sp.csr_matrix((increment, (y, one))).toarray()[:, 1]
        pAB = sp.csr_matrix((increment, (x, y)))
        (A, B) = np.nonzero(pAB)
        AB = pAB.data
        ami = np.sum(np.multiply(AB, np.log2(np.divide(AB, np.multiply(pA[A], pB[B])))))

        if to_matlab:
            import matlab
            return ami
        else:
            return ami
    else:
        raise ValueError('Invalid input, read documentation for input options.')

x = np.arange(0,1000,0.1)   # start,stop,step
dx = np.sin(x/5)


#x_1 = dx[:-2]
#y_1 = dx[1:-1]
#z_1 = dx[2:]
#x_3 = dx[:-6]
#y_3 = dx[3:-3]
#z_3 = dx[6:]
#
#x_12 = dx[:-24]
#y_12 = dx[12:-12]
#z_12 = dx[24:]
#
#x_200 = dx[:-400]
#y_200 = dx[200:-200]
#z_200 = dx[400:]
#
#
#x_1000 = dx[:-2000]
#y_1000 = dx[1000:-1000]
#z_1000 = dx[2000:]
#
#print(len(x_1))
#print(len(y_1))
#print(len(z_1))
#print(len(x_3))
#print(len(y_3))
#print(len(z_3))
#ax = plt.axes(projection='3d')
#ax.plot3D(x_1, y_1, z_1, 'red')
#plt.show()
#
#ax = plt.axes(projection='3d')
#ax.plot3D(x_3, y_3, z_3, 'red')
#plt.show()
#
#ax = plt.axes(projection='3d')
#ax.plot3D(x_12, y_12, z_12, 'red')
#plt.show()
#
#ax = plt.axes(projection='3d')
#ax.plot3D(x_200, y_200, z_200, 'red')
#plt.show()
#
#ax = plt.axes(projection='3d')
#ax.plot3D(x_1000, y_1000, z_1000, 'red')
#plt.show()

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


# Set other parameters
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
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)


x_1 = xs[:-2]
y_1 = xs[1:-1]
z_1 = xs[2:]
x_3 = xs[:-6]
y_3 = xs[3:-3]
z_3 = xs[6:]

x_12 = xs[:-24]
y_12 = xs[12:-12]
z_12 = xs[24:]

x_200 = xs[:-400]
y_200 = xs[200:-200]
z_200 = xs[400:]


x_1000 = xs[:-2000]
y_1000 = xs[1000:-1000]
z_1000 = xs[2000:]

print(len(x_1))
print(len(y_1))
print(len(z_1))
print(len(x_3))
print(len(y_3))
print(len(z_3))
#ax = plt.axes(projection='3d')
#ax.plot3D(x_1, y_1, z_1, 'red')
#plt.show()
#
#ax = plt.axes(projection='3d')
#ax.plot3D(x_3, y_3, z_3, 'red')
#plt.show()
#
#ax = plt.axes(projection='3d')
#ax.plot3D(x_12, y_12, z_12, 'red')
#plt.show()
#
#ax = plt.axes(projection='3d')
#ax.plot3D(x_200, y_200, z_200, 'red')
#plt.show()
#
#ax = plt.axes(projection='3d')
#ax.plot3D(x_1000, y_1000, z_1000, 'red')
#plt.show()

x_1 = xs[:-2]
y_1 = xs[1:-1]
z_1 = xs[2:]

x_30 = xs[:-60]
y_30 = xs[30:-30]
z_30 = xs[60:]

x_11 = xs[:-22]
y_11 = xs[11:-11]
z_11 = xs[22:]

ax = plt.axes(projection='3d')
ax.plot3D(x_1, y_1, z_1, 'red')
plt.show()

ax = plt.axes(projection='3d')
ax.plot3D(x_30, y_30, z_30, 'red')
plt.show()

ax = plt.axes(projection='3d')
ax.plot3D(x_11, y_11, z_11, 'red')
plt.show()