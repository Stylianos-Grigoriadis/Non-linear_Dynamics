import lib
import colorednoise as cn
import pandas as pd
import matplotlib.pyplot as plt
import nolds
import Nonlinear_Methods as nm
import numpy as np

data = pd.read_excel(
    r'C:\Users\vmylo\OneDrive - University of Nebraska at Omaha\AUTH drive\LabProjects\nonlinear dynamics\algorithm_tests\data.xlsx')
white = np.array(data['White'])
pink = np.array(data['Pink'])
red = np.array(data['Red'])

time_lag = 24
ed = 4
print('FS:1')
LyE_W = nm.LyE_W(pink, 1, time_lag, ed, 10)[1]
print('LyE, Wolf (pink): ', LyE_W)

LyE_W = nm.LyE_W(white, 1, time_lag, ed, 10)[1]
print('LyE, Wolf (white): ', LyE_W)

LyE_W = nm.LyE_W(red, 1, time_lag, ed, 10)[1]
print('LyE, Wolf (red): ', LyE_W)


print('evolve:10')
LyE_W = nm.LyE_W(pink, 1, time_lag, ed, 10)[1]
print('LyE, Wolf (pink): ', LyE_W)

LyE_W = nm.LyE_W(white, 1, time_lag, ed, 10)[1]
print('LyE, Wolf (white): ', LyE_W)

LyE_W = nm.LyE_W(red, 1, time_lag, ed, 10)[1]
print('LyE, Wolf (red): ', LyE_W)


lyew = []
for i in range(1,100):
    lyew.append(nm.LyE_W(pink, 1, time_lag, ed, i)[1])
plt.plot(lyew)

pink = pink[:1000]
lyew = []
for i in range(1,100):
    lyew.append(nm.LyE_W(pink, 1, time_lag, ed, i)[1])
plt.plot(lyew)
plt.show()


# print(time_lag)
# lyew = []
# for i in range(1,20):
#     lyew.append(nm.LyE_W(pink, 1, time_lag, i, 50)[1])
# plt.plot(lyew)
# plt.show()
