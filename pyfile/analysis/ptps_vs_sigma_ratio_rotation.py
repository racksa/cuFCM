import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

boxsize=200
phi_array = np.array([0.0005*4**j for j in range(5)])
N_array = phi_array*(boxsize/0.5)**3/(4./3.*np.pi)

sigma_ratio_array = np.array([11.81635901, 10.12830772,  8.86226925,  7.87757267,  7.0898154 ,
  6.44528673,  5.9081795 ,  5.45370416,  5.06415386,  4.7265436 ,
  4.43113463,  4.17047965,  3.93878634,  3.73148179,  3.5449077 ,
  3.37610257,  3.22264337,  3.08252844,  2.95408975,  2.83592616,
  2.72685208,  2.62585756,  2.53207693,  2.44476393,  2.3632718 ,
  2.28703723,  2.21556731,  2.14842891,  2.08523982,  2.02566154,
  1.96939317,  1.91616633,  1.8657409 ,  1.81790139,  1.77245385,
  1.72922327,  1.68805129,  1.64879428,  1.61132168,  1.57551453,
  1.54126422,  1.50847136,  1.47704488,  1.4469011 ,  1.41796308,
  1.39015988,  1.36342604,  1.33770102,  1.31292878]
)

ptps_array = np.array([[4629360.28894868, 5217504.1492784 , 3969610.83367544, 5338933.89059344,
 4980765.46912695, 5068775.83655703, 6643244.13639566, 2557158.6382352 ,
 4532912.42144989, 4517071.32466857, 3911498.4433885 , 1392877.52560477,
 2780426.72762082, 1073980.814842  , 2028309.777255  , 1777805.705083  ,
 1592689.34947637,  627865.42194956, 1271723.2607657 , 1141971.29449824,
  820883.25275661,  853431.82600224,  826134.35308673,  316910.20227658,
  727710.93519286,  263641.16087702,  534395.73546747,  501537.63731718,
  270133.5992588 ,  414578.3528122 ,  406632.81519899,  149762.14305529,
  274681.80922505,  104062.66700396,  276471.64419377,  107541.18669778,
  247771.06589238,   94063.18351728,  231651.42692002,  227414.94788122,
  112981.19800003,   71147.13225087,  171905.4432298 ,  167471.57817244,
   92314.53407082,   47751.47977079,   65649.3072275 ,   47157.43861832,
  124346.80893109],
                        [2063807.35546696, 2886692.79682433, 3536411.48613304, 4786347.54550296,
 5784068.02247225, 6553766.1045231 , 6611550.57813474, 5445822.46943107,
 7625162.5122586 , 7399165.57500502, 7390754.3614811 , 4310663.45028002,
 6889764.22337113, 3743134.05557434, 6080982.92341707, 5615867.11730338,
 5239190.10902906, 2377847.1211685 , 4449016.19915379, 3945068.30957367,
 2854268.55923749, 2923823.84910975, 2750150.07609556, 1158151.3089797 ,
 2387071.67542637,  963715.20933533, 1778228.06850109, 1683681.45322146,
  979127.10067575, 1405922.1081332 , 1388968.4351389 ,  554970.03939112,
  951416.33595268,  392759.1041077 ,  941572.98800671,  405384.12523885,
  865912.86689788,  357241.52073099,  813802.33474475,  801072.32589647,
  423389.17233839,  271934.89011764,  618026.04620684,  599132.54501239,
  347353.53933557,  184307.77643467,  250688.16802219,  181825.46290756,
  454374.45260793],
                        [863543.39911396, 1255251.22328714, 1626127.88259205, 2187422.93553124,
 2771972.75424964, 3308709.97666876, 3978254.50869387, 4607362.12969217,
 5348686.39272827, 5641248.35753967, 6224822.90091109, 5871445.93302814,
 7018347.72832454, 6116347.20516009, 7670575.06407312, 8063393.59163724,
 7720902.90915637, 5450484.33637912, 7802027.77401886, 7567465.16229529,
 6766219.7186342 , 7784576.81911861, 6913035.33888465, 3773696.75013793,
 6558050.79471364, 3276062.76381981, 5331510.65429757, 5185785.44093813,
 3315890.74786995, 4493493.93692639, 4295327.95670585, 1924832.17756266,
 2950269.03132311, 1389321.28762338, 2944695.23620922, 1441694.80937971,
 2717623.30801658, 1273331.16579859, 2510915.28131167, 2446070.08812296,
 1438317.95287452,  969710.6441511 , 1875893.01746669, 1811606.55397544,
 1178799.12936865,  673898.50977073,  881549.3516541 ,  661055.67069812,
 1427895.70602101],
                        [328568.22282267,  482627.43400496,  680333.90258002,  903588.43385071,
 1162952.16588165, 1455627.78505227, 1766490.3547263 , 2089024.40014408,
 2573979.47620376, 2899516.8667856 , 3260696.60248596, 3620016.94493976,
 4148168.07053505, 4512135.91180218, 4984998.80606279, 5311334.4999107 ,
 5615573.5534893 , 5459705.32824976, 6301064.86724246, 6818658.84447509,
 6939795.10612036, 7298374.53484194, 7895997.37469227, 6474797.08776294,
 8257212.30778481, 6308108.40869098, 8061879.47554244, 8054087.84621353,
 6741979.12525523, 7892057.741512  , 7943247.363247  , 5183534.50169589,
 7083554.33688286, 4292350.35379278, 8486102.90282576, 4386316.4489177 ,
 7004471.96373146, 4075956.78772025, 6846891.0289221 , 6838837.62294362,
 4662809.11830513, 3376488.52182574, 5507973.11730868, 5688690.71907416,
 4024692.62022658, 2403444.12235624, 3022475.24070017, 2327179.03004276,
 4267989.17767775],
                        [110385.7757527 ,  157758.87522205,  231202.35573968,  312901.6988025 ,
  400626.57115278,  524075.59656393,  624032.41145266,  762563.61370903,
  888769.44073373, 1052606.50893227, 1227101.00502069, 1407802.96276875,
 1631054.33307993, 1807942.8084761 , 2032789.77176817, 2267038.92707173,
 2559005.06068523, 2756228.31043111, 2979495.60333219, 3198258.29312524,
 3471890.0368289 , 3718206.20893163, 3935475.89960074, 4169122.83921125,
 4630409.96907535, 4681231.98889772, 5058547.07044789, 5174153.83306381,
 5240627.04609936, 5593254.26128872, 5827880.78766259, 5528289.52583058,
 6147316.57331095, 5615920.64305083, 6713196.55728194, 6016311.8545088 ,
 7084159.4645221 , 6142096.12560473, 7509892.35102241, 7769058.20555788,
 7022683.93395981, 6244828.89198532, 7836172.95992927, 7927560.14798799,
 6973237.20555496, 5509798.06708904, 6308977.45237245, 5561203.56616325,
 7835727.75654492]]
)

pop_index = [7, 11, 13, 17, 23, 25, 28, 31, 33, 35, 37, 40, 41, 44, 45, 46, 47]
sigma_ratio_array = np.delete(sigma_ratio_array, pop_index)
ptps_array = np.delete(ptps_array, pop_index, axis=1)

def plot_ptps_vs_npts(sigma_ratio_array, ptps_array):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    # plot data
    for i, phi in enumerate(phi_array):
        ax.plot(sigma_ratio_array, ptps_array[i], marker='+', label=rf"$N$={int(N_array[i])},$\phi$={phi*100}%")
    ax.legend()
    
    # adding title and labels
    ax.set_ylabel(r"PTPS")
    ax.set_xlabel(r"$\Sigma/\sigma$")
    ax.set_xlim((1, max(sigma_ratio_array)))
    ax.set_ylim((0, 1e7))
    ax.set_title(r"PTPS vs. $\Sigma/\sigma$")
    ax.set_ylabel('PTPS')
    plt.savefig('img/ptps_vs_sigmaratio_rotation.eps', format='eps')
    plt.show()

plot_ptps_vs_npts(sigma_ratio_array, ptps_array)
