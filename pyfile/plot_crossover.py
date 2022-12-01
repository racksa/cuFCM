import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def inv_f(x, a):
    return a/x**.5

aL_array = np.array([0.004, 0.0049, 0.0059, 0.0068, 0.0078, 0.0088, 0.0097, 0.0107, 0.0116, 0.0126, 0.0135])
crossover_array = np.array([0.180, 0.183, 0.160, 0.155, 0.139, 0.120, 0.131, 0.124, 0.103, 0.1075, 0.089])

p0 = curve_fit(inv_f, aL_array, crossover_array)
theory_x = list()
theory_y = list()
for sec in range(2):
    if sec == 0:
        x_array = np.linspace(1e-10, 0.015, 50)
    else:
        x_array = np.linspace(0.004, 0.0135, 50)
    theory_x.append(x_array)
    theory_y.append(inv_f(x_array, p0[0]))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for sec in range(2):
    if sec == 1:
        linestyle = 'solid'
        label = 'Empirical fit'
    else:
        linestyle = 'dotted'
        label = 'Prediction'
    ax.plot(theory_x[sec], theory_y[sec], linestyle=linestyle, color='blue', label=label)
ax.fill_between(theory_x[0], 0, theory_y[0], color='mediumaquamarine', alpha=0.5)
ax.fill_between(theory_x[0], theory_y[0], 1, color='beige', alpha=0.5)
ax.scatter(aL_array, crossover_array, marker = '+', s=100, color='red', label='Data', zorder=10)
ax.set_xlabel('a/L')
ax.set_ylabel(r'$\phi_c$')
ax.set_title(r"Crossover volume fraction vs. aspect ratio")
ax.set_xlim((0, 0.015))
ax.set_ylim((0, 0.25))
ax.legend()
ax.annotate('FCM region', (0.004, 0.22), weight='bold', size=25)
ax.annotate('FFCM region', (0.0015, 0.06), weight='bold', size=25)

plt.savefig('img/r_crossover.eps', format='eps')
plt.show()