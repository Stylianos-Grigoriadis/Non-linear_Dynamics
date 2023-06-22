import matplotlib.pyplot as plt
import numpy as np
import statistics as st
from scipy.stats import norm

x = np.arange(0,1000,0.1)   # start,stop,step
dx = np.sin(x/5)
print(type(dx))
plt.plot(x,dx)
plt.show()

derivative_periodic_signal = np.gradient(np.sin(x), dx)
plt.plot(dx,np.gradient(np.sin(x),dx), '-*', label='approx')
plt.show()

rand_data = np.random.normal(loc=0, scale=1, size=500)
print(rand_data)
st.mean(rand_data)
print(st.mean(rand_data))
print(st.stdev(rand_data))

h = plt.hist(rand_data, bins=30, histtype='bar', density=1, ec='k')
plt.plot(h[1], norm.pdf(h[1], st.mean(rand_data), st.stdev(rand_data)), color='r', linewidth=2)
plt.title(r'Mean=%.4f, Std Dev=%.4f' %(st.mean(rand_data), st.stdev(rand_data)))
plt.show()
