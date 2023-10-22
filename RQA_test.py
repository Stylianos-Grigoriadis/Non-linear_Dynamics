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
pink = data['Red']
red = data['Red']

time_lag = 24
ed = 4

# print('RQA:', nm.RQA(white,TYPE='RQA',EMB=ed,DEL=time_lag,ZSCORE=0, NORM='non', SETPARA = 'radius', SETVALUE=2.5,LINELENGTH = None,PLOTOPTION=1,nargout=2))
# print('RQA:', nm.RQA(pink,TYPE='RQA',EMB=ed,DEL=time_lag,ZSCORE=0, NORM='non', SETPARA = 'radius', SETVALUE=2.5,LINELENGTH = None,PLOTOPTION=1,nargout=2))
# print('RQA:', nm.RQA(red,TYPE='RQA',EMB=ed,DEL=time_lag,ZSCORE=0, NORM='non', SETPARA = 'radius', SETVALUE=2.5,LINELENGTH = None,PLOTOPTION=1,nargout=2))
print('RQA:', nm.RQA(np.array(pink,red),TYPE='cRQA',EMB=ed,DEL=time_lag,ZSCORE=0, NORM='non', SETPARA = 'radius', SETVALUE=2.5,LINELENGTH = None,PLOTOPTION=1,nargout=2))


