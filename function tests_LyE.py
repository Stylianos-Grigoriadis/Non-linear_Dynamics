import lib
import colorednoise as cn
import pandas as pd
import matplotlib.pyplot as plt
import nolds
import Nonlinear_Methods as nm
import numpy as np

data = pd.read_excel(
    r'C:\Users\Βασίλης\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\BiomechLabProjects\nonlinear dynamics\algorithm_tests\data.xlsx')
white = data['White']
pink = data['Pink']
red = data['Red']

time_lag = 24
ed = 4
print('FS:1')
LyE_W = nm.LyE_W(pink, 1, time_lag, ed, 10)[1]
print('LyE, Wolf (pink): ', LyE_W)

LyE_W = nm.LyE_W(white, 1, time_lag, ed, 10)[1]
print('LyE, Wolf (white): ', LyE_W)

LyE_W = nm.LyE_W(red, 1, time_lag, ed, 10)[1]
print('LyE, Wolf (red): ', LyE_W)


print('FS:1')
LyE_W = nm.LyE_W(pink, 25, time_lag, ed, 10)[1]
print('LyE, Wolf (pink): ', LyE_W)

LyE_W = nm.LyE_W(white, 25, time_lag, ed, 10)[1]
print('LyE, Wolf (white): ', LyE_W)

LyE_W = nm.LyE_W(red, 25, time_lag, ed, 10)[1]
print('LyE, Wolf (red): ', LyE_W)
