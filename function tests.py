import lib
import colorednoise as cn
import pandas as pd
import matplotlib.pyplot as plt
import nolds
import Nonlinear_Methods as nm

data = pd.read_excel(
    r'C:\Users\Βασίλης\OneDrive - Αριστοτέλειο Πανεπιστήμιο Θεσσαλονίκης\BiomechLabProjects\nonlinear dynamics\algorithm_tests\data.xlsx')
white = data['White']
pink = data['Pink']
red = data['Red']
print('PINK:')
time_lag = (nm.Time_delay(pink, 30, 'pink'))
print(time_lag)
ed = nm.Culculation_of_embending_dimensions(pink, time_lag, 10, 0, 'pink')[1]
print(ed)
LyE_W = nm.LyE_W(pink, 1, time_lag, ed, 10)[1]
print('Haurst epxonent a (UNO, Matlab): ', 0.9280)
print('Haurst epxonent a (fathon): ', lib.DFA(pink))
print('Haurst epxonent a (nolds): ', nolds.dfa(pink))
print()
print('LyE, Wolf (UNO): ', LyE_W)
print('LyE, Eckmann (nolds)', max(nolds.lyap_e(pink, emb_dim=ed, tau=time_lag)))
print('LyE, Rosenstein (nolds)', nolds.lyap_r(pink, emb_dim=ed, tau=time_lag))
print()
print('Sample Entropy (UNO): ', nm.Ent_Samp(pink, 2, 0.2))
print('Sample Entropy (nolds): ', nolds.sampen(pink))
print('approximate Entropy (UNO): ', nm.Ent_Ap(pink, ed, 0.2))

print()

print('WHITE:')
time_lag = (nm.Time_delay(white, 30, 'pink'))
print(time_lag)
ed = nm.Culculation_of_embending_dimensions(white, time_lag, 10, 0, 'pink')[1]
print(ed)
LyE_W = nm.LyE_W(white, 1, time_lag, ed, 10)[1]
print('Haurst epxonent a (UNO, Matlab): ', 0.55)
print('Haurst epxonent a (fathon): ', lib.DFA(white))
print('Haurst epxonent a (nolds): ', nolds.dfa(white))
print()
print('LyE, Wolf (UNO): ', LyE_W)
print('LyE, Eckmann (nolds)', max(nolds.lyap_e(white, emb_dim=ed, tau=time_lag)))
print('LyE, Rosenstein (nolds)', nolds.lyap_r(white, emb_dim=ed, tau=time_lag))
print()
print('Sample Entropy (UNO): ', nm.Ent_Samp(white, 2, 0.2))
print('Sample Entropy (nolds): ', nolds.sampen(white))
print('approximate Entropy (UNO): ', nm.Ent_Ap(white, ed, 0.2))
print()

print('RED:')
time_lag = (nm.Time_delay(red, 30, 'pink'))
print(time_lag)
ed = nm.Culculation_of_embending_dimensions(red, time_lag, 10, 0, 'pink')[1]
print(ed)
LyE_W = nm.LyE_W(red, 1, time_lag, ed, 10)[1]
print('Haurst epxonent a (UNO, Matlab): ', 1.5185)
print('Haurst epxonent a (fathon): ', lib.DFA(red))
print('Haurst epxonent a (nolds): ', nolds.dfa(red))
print()
print('LyE, Wolf (UNO): ', LyE_W)
print('LyE, Eckmann (nolds)', max(nolds.lyap_e(red, emb_dim=ed, tau=time_lag)))
print('LyE, Rosenstein (nolds)', nolds.lyap_r(red, emb_dim=ed, tau=time_lag))
print()
print('Sample Entropy (UNO): ', nm.Ent_Samp(red, 2, 0.2))
print('Sample Entropy (nolds): ', nolds.sampen(red))
print('approximate Entropy (UNO): ', nm.Ent_Ap(red, ed, 0.2))
print()

fig, (ax, ax1, ax2) = plt.subplots(3)
ax.plot(white)
ax1.plot(pink)
ax2.plot(red)
plt.show()
