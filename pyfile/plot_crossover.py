import numpy as np
import matplotlib.pyplot as plt

aL_array = np.array([0.004, 0.008, 0.012])
crossover_array = np.array([0.18, 0.1471, 0.0955])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(aL_array, crossover_array)
ax.set_xlabel('r/L')
ax.set_ylabel(r'$\phi$')

plt.savefig('img/r_crossover.eps', format='eps')
plt.show()