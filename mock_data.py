import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Nonlinear_Methods as nm
import lib

pink = pd.read_excel(
    r'C:\Users\vmylo\OneDrive - University of Nebraska at Omaha\AUTH drive\LabProjects\nonlinear dynamics\algorithm_tests\data.xlsx')['Pink']
pink = np.array(pink[:750])

deltas_cols = ['Time',	'CHANNEL_1L',	'CHANNEL_2L', 'CHANNEL_3L',	'CHANNEL_4L', 'Empty', 'CHANNEL_1R',	'CHANNEL_2R'	,'CHANNEL_3R',	'CHANNEL_4R']

df = pd.read_csv(r"C:\Users\vmylo\OneDrive - University of Nebraska at Omaha\AUTH drive\Kinvent\TKA\raw_data\subject1\pre\stance evaluation_Μαλουτα Παρθένα  21Φεβ22_09_43_34.csv",delimiter=',',
                                 decimal='.',
                                 thousands=',',
                                 skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] ,
                                 header=None,
                                 names=deltas_cols,
                                    index_col=False)
X = 120
Y = 260

copx,copy = lib.compute_cop(X,Y,df)
copx = np.array(copx)
copy = np.array(copy)
copx = copx - np.mean(copx)
copy = copy - np.mean(copy)

x = np.arange(0,7.5,0.01)
sine = np.sin(x)
rand = np.random.rand(750,1).transpose()[0]
rand = rand/10
sine = sine + rand


time_lag = (nm.Time_delay(pink, 30, 'pink'))
ed = nm.Calculation_of_embending_dimensions(pink, time_lag, 10, 0, 'pink')[1]

lor = nm.lorenz(1,2,3)

print('LyE Sine:', nm.LyE_W(sine, 1, 6, 4, 50)[1])
dt = 0.01
num_steps = 7500
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)
xs[0], ys[0], zs[0] = (0., 1., 1.05)
x_dot, y_dot, z_dot = nm.lorenz(xs[0], ys[0], zs[0])
for i in range(num_steps):
    x_dot, y_dot, z_dot = nm.lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)

print('LyE Lorenz:', nm.LyE_W(xs, 1, 6, 4, 50)[1])
white = pd.read_excel(
    r'C:\Users\vmylo\OneDrive - University of Nebraska at Omaha\AUTH drive\LabProjects\nonlinear dynamics\algorithm_tests\data.xlsx')['White']
white = np.array(white[:750])
print('LyE White:', nm.LyE_W(white, 1, 6, 4, 50)[1])