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

ptps_array = np.array([[7165974.05277624, 5176175.63355468, 5604260.96972276, 6567115.42098657,
 6259987.38004901, 6127671.1801322 , 6485600.76071453, 2631642.40806132,
 5310171.2823935 , 4640524.86103939, 3614845.52083777, 1487883.07291971,
 2793022.38001046, 1074134.84969283, 2023795.96194554, 1780607.02406232,
 1594734.18313452,  626979.57371298, 1273393.75521968, 1137505.82601328,
  818932.24699828,  861628.48799371,  833769.00920213,  317908.58681955,
  728089.63918626,  264456.4811744 ,  552221.39169333,  508334.71968059,
  273319.74904245,  422812.73696387,  408834.33245044,  149943.78334919,
  276413.72103382,  104127.23614909,  277322.47132477,  107656.91675608,
  247868.42563801,   94467.97796481,  232339.57650265,  227866.39911729,
  113300.67453873,   71502.40317902,  173039.73435377,  167529.97567208,
   92144.55320024,   47733.89891661,   65618.36482711,   47193.0367567 ,
  124404.88734184],
                        [3289745.66563601, 4558711.85843929, 5463514.58629839, 7057402.45462818,
 8275042.17815294, 8891559.46377701, 8781049.92126165, 6464257.98644827,
 9167604.218232  , 8776585.81419729, 8454051.48181758, 4579570.35336293,
 7490617.42376887, 3794359.3099449 , 6420360.59242886, 5845365.71555372,
 5412274.81327059, 2388497.19386559, 4472871.04248913, 3960233.07533994,
 2856584.62527952, 2941196.28733766, 2770288.27969604, 1163457.45272229,
 2417078.78109051,  971070.9396288 , 1789530.33686414, 1681210.89879338,
  978904.39720511, 1418255.05012882, 1382987.58266606,  554686.45339396,
  959043.32313517,  394058.87606109,  940347.03320003,  405057.40489396,
  872028.89621295,  356989.02069209,  813701.5658902 ,  801896.60868878,
  423050.60229021,  272022.49889658,  616900.86705692,  599263.9077294 ,
  347638.8160462 ,  184634.85386644,  250640.01091244,  181783.71004508,
  454310.88352368],
                        [1478699.01585401, 2120831.34228861, 2733524.82692671, 3599905.40050604,
 4469931.18288064, 5250806.221924  , 6062496.83805798, 6712271.82092408,
 7824587.88562006, 8153944.98705495, 8707713.48252972, 7359054.80427732,
 9143319.64665479, 7525138.69145208, 9520834.76136767, 9573586.51943776,
 9318491.445788  , 6100802.24326643, 8812240.0955122 , 8782789.95673633,
 7292915.91338954, 8101145.34805498, 7644845.84280626, 3940382.65200035,
 7035763.64485137, 3379942.2287875 , 5524617.15709339, 5242671.39048959,
 3322803.73628525, 4491375.51544441, 4291791.24261103, 1931890.85061677,
 2958608.52693101, 1390042.5507912 , 2943508.8561736 , 1442081.59989769,
 2703620.94539805, 1271442.39277906, 2499734.6473284 , 2443148.76963084,
 1436378.91964986,  970697.99245074, 1873351.16207392, 1816832.43299325,
 1176627.32767266,  673901.96515503,  881087.1821458 ,  661688.18890102,
 1433279.95617257],
                        [571946.56957813,   836525.00488135,  1170487.80998722,
  1540872.63116948,  1967122.87232617,  2456038.03289098,
  2937442.15436737,  3400384.07440461,  4137815.52744426,
  4619356.56495522,  5129442.8337952 ,  5564112.94356179,
  6308977.74722152,  6626220.02975899,  7422695.40628275,
  7749962.98770241,  8071092.93040846,  7465681.46906217,
  8870093.26312822,  9276166.704359  ,  9250138.34824026,
  9646676.74893283, 10115680.36698599,  7805063.61809381,
 10348084.81601636,  7373113.09437367,  9777108.80356335,
  9749667.28258108,  7820303.04048164,  9343637.0321617 ,
  9317708.62614572,  5702300.50001539,  8075249.39355897,
  4615476.83779661,  8957407.87119234,  4705399.4541262 ,
  7800263.3043354 ,  4320996.87398669,  7536992.96297798,
  7497815.09316314,  4907921.82025243,  3499564.23921378,
  5743777.09413325,  5829335.01969529,  4057755.01425376,
  2414143.5592875 ,  3041539.28863055,  2338165.7935502 ,
  4296735.14462065],
                        [190236.12651448,  272099.27714083,  398816.17792449,  540030.08132915,
  694287.69847708,  901857.03938894, 1071379.92646547, 1297864.99964904,
 1511646.43970295, 1777041.16192856, 2065519.84695364, 2351301.359721  ,
 2706945.56226789, 2957845.28080833, 3311746.05174411, 3655338.71368719,
 4096926.35678012, 4336617.84320686, 4687812.17254654, 5025295.23267418,
 5368329.06137833, 5719432.08412199, 5986166.79847368, 6178661.42624355,
 6892180.43004116, 6759103.41095222, 7346851.82067866, 7460961.27962165,
 7405075.11530205, 7929964.89470134, 8156328.14572243, 7439540.12515593,
 8412543.01024159, 7286947.7082571 , 9014152.39833207, 7663985.89105066,
 9276169.0542851 , 7656440.76187203, 9682026.04524637, 9946709.66865171,
 8603119.78593719, 7386436.04097498, 9719607.22686974, 9810948.25763796,
 8327474.94661452, 6301249.73978321, 7360868.98297199, 6311513.77705935,
 9346469.49677231]]
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
    ax.set_ylim((0, 1.2e7))
    # ax.set_title(r"PTPS vs. $\Sigma/\sigma$")
    ax.set_ylabel('PTPS')
    plt.savefig('img/ptps_vs_sigmaratio_linear.eps', bbox_inches = 'tight', format='eps')
    plt.show()

plot_ptps_vs_npts(sigma_ratio_array, ptps_array)
