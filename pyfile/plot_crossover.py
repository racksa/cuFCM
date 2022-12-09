import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as colors

def inv_f(x, a):
    return a/x**.5

def N_from_al_and_phi(al, phi):
    return phi/(4./3.*np.pi*al**3)

aL_array = np.array([0.004, 0.0049, 0.0059, 0.0068, 0.0078, 0.0088, 0.0097, 0.0107, 0.0116, 0.0126, 0.0135])
crossover_array = np.array([0.180, 0.183, 0.160, 0.155, 0.139, 0.120, 0.131, 0.124, 0.103, 0.1075, 0.089])

aL_domain, phic_domain = np.meshgrid(np.linspace(5e-4, 0.014, 100), np.linspace(0.001, 0.25, 100) )
N_domain = N_from_al_and_phi(aL_domain, phic_domain)
print(aL_domain)
print(N_domain)

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

fig, ax = plt.subplots()
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
# CS = ax.contour(aL_domain, phic_domain, N_domain, norm=colors.LogNorm(vmin=100, vmax=1e7))
# ax.clabel(CS, inline=True, fontsize=10)
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

# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
# X, Y = np.meshgrid(x, y)
# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# Z = (Z1 - Z2) * 2

# fig, ax = plt.subplots()
# CS = ax.contour(X, Y, Z)
# ax.clabel(CS, inline=True, fontsize=10)
# ax.set_title('Simplest default with labels')
# plt.show()