import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as colors
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

def inv_f(x, a):
    return a/x**.5

def exp(x, a, b):
    return np.exp(-b*x)

def linear(x, k, c):
    return k*x+c

def N_from_al_and_phi(al, phi):
    return phi/(4./3.*np.pi*al**3)



# aL_array = np.array([0.004, 0.0049, 0.0059, 0.0068, 0.0078, 0.0088, 0.0097, 0.0107, 0.0116, 0.0126, 0.0135])
# crossover_array = np.array([0.180, 0.183, 0.160, 0.155, 0.139, 0.120, 0.131, 0.124, 0.103, 0.1075, 0.089])

# 10^-4
aL_array = np.linspace(0.004, 0.014, 11)
crossover_array = np.array([0.1796988332018473, 0.18341836874267936, 0.16029990375180314,
                            0.1555956484307304, 0.1393131236584624, 0.11949121802627433,
                            0.13161303422224319, 0.11153336982307753, 0.10390989284238716,
                            0.10763130372030884, 0.08996120945668212])

# 10^-6
aL_array = np.linspace(0.0039788735772973835, 0.014483099821362476, 11)
crossover_array = np.array([0.1930290404505306, 0.14157834543805098, 0.12587200178191077,
                            0.1233225491615936, 0.1052276484392654, 0.10186029460524607,
                            0.10437436862302342, 0.09806258691269104, 0.08517675667040094,
                            0.096326320026101, 0.09298367081611289])


aL_domain, phic_domain = np.meshgrid(np.linspace(5e-4, 0.014, 100), np.linspace(0.001, 0.25, 100) )
N_domain = N_from_al_and_phi(aL_domain, phic_domain)
# print(aL_domain)
# print(N_domain)

# p0 = curve_fit(inv_f, aL_array, crossover_array)
p0 = curve_fit(linear, aL_array, crossover_array)
print(p0)
theory_x = list()
theory_y = list()
for sec in range(2):
    if sec == 0:
        x_array = np.linspace(0, 0.016, 50)
    else:
        x_array = np.linspace(0.004, 0.014, 50)
    theory_x.append(x_array)
    # theory_y.append(inv_f(x_array, p0[0]))
    theory_y.append(linear(x_array, p0[0][0], p0[0][1]))


fig = plt.figure(figsize=(4.8, 3.6))
ax = fig.add_subplot(1,1,1)
for sec in range(2):
    if sec == 1:
        linestyle = 'solid'
        label = 'Empirical fit'
    else:
        linestyle = 'dotted'
        label = 'Prediction'
    ax.plot(theory_x[sec], theory_y[sec], linestyle=linestyle, color='black', label=label)
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
ax.fill_between(theory_x[0], 0, theory_y[0], color='grey', alpha=0.5)
ax.fill_between(theory_x[0], theory_y[0], 1, color='white', alpha=0.5)
ax.scatter(aL_array, crossover_array, marker = '+', color='black', label='Data', zorder=10)
# CS = ax.contour(aL_domain, phic_domain, N_domain, norm=colors.LogNorm(vmin=100, vmax=1e7))
# ax.clabel(CS, inline=True, fontsize=10)
ax.set_xlabel(r'$a/L$')
ax.set_ylabel(r'$\phi_c$')
# ax.set_title(r"Crossover volume fraction vs. aspect ratio")
ax.set_xlim((0, 0.016))
ax.set_ylim((0, 0.25))
ax.legend()
ax.annotate('FCM region', (0.004, 0.22), size=12)
ax.annotate('FFCM region', (0.0015, 0.06), size=12)
ax.annotate(rf'fit: $y={p0[0][0]:.2f}x+{p0[0][1]:.2f}$', (0.008, 0.16), size=12)

plt.savefig('img/r_crossover.pdf', bbox_inches = 'tight', format='pdf')
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