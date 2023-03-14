import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

W = 1./(6*np.pi)

La_ratio_array = np.array([
    5, 10, 20, 80, 100, 200
]
)

error_array = np.array([
    0.02472110151830464, 0.012620319656578866, 0.00634297303229032, 0.001588309445393014,
    0.0012706902479938536, 0.000635386442719013
]
)/W

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

# plot data
ax.plot(La_ratio_array, error_array, marker='+')

# adding title and labels
# ax.set_xlim((1, max(sigma_ratio_array)))
# ax.set_ylim((0, 1.2e7))
# ax.set_title(r"PTPS vs. $\Sigma/\sigma$")
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel(r'$\frac{\sum_{\mathbf{x}\in \mathbf{X}}|u_{fcm}(\mathbf{x}) - u_{expr}(\mathbf{x})|}{W|\mathbf{X}|}$', fontsize=18)
ax.set_xlabel(r"$L/a$", fontsize=18)
plt.tight_layout()
plt.savefig('img/flow_error.eps', format='eps')
plt.show()
