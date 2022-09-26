import sys
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from python import util

def plot_3Dheatmap(alpha_array, beta_array, eta_array ):
    colo = [alpha_array + beta_array + eta_array]

    # creating figures
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # setting color bar
    color_map = cm.ScalarMappable(cmap=cm.Greens_r)
    color_map.set_array(colo)
    
    # creating the heatmap
    img = ax.scatter(alpha_array, beta_array, eta_array, marker='s',
                    s=200, color='green')
    plt.colorbar(color_map)
    
    # adding title and labels
    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    # displaying plot
    plt.show()

def run():
    info_file_name = "simulation_info"
    par_dict = util.read_info(info_file_name)
    save_directory = "data/simulation/test/"

    il, jl, kl = 6, 6, 6
    error_array = np.zeros((il, jl, kl))
    alpha_array = np.linspace(0.8, 1.0, il)
    beta_array = np.linspace(9.5, 14, jl)
    eta_array = np.linspace(5.5, 8.5, kl)

    # Begin custom massive loop
    for i in range(il):
        for j in range(jl):
            for k in range(kl):

                npts = 256
                par_dict['N']=          500000.0
                par_dict['rh']=         0.02609300415934458
                par_dict['alpha']=      alpha_array[i]
                par_dict['beta']=       9.706
                par_dict['eta']=        5.573
                par_dict['nx']=         npts
                par_dict['ny']=         npts
                par_dict['nz']=         npts
                par_dict['repeat']=     1
                par_dict['prompt']=     -1

                # alpha_array[i][j][k] = par_dict['alpha']
                # beta_array[i][j][k] = par_dict['beta']
                # eta_array[i][j][k] = par_dict['eta']

                if(sys.argv[1] == 'run'):
                    for key in par_dict:
                        util.replace(key, str(par_dict[key]), info_file_name)

                    subprocess.call("./bin/CUFCM", shell=True)
                    util.savefile(par_dict, save_directory, 2)

                    sim_dict = util.read_scalar(save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat")
                    print("(", str(npts), str(par_dict['alpha']), par_dict['beta'], par_dict['eta'], ") "
                          "Verror=", str(sim_dict["Verror"]),\
                          "Werror=", str(sim_dict["Werror"]),\
                          "time_compute=", str(sim_dict["time_compute"]))

                if(sys.argv[1] == 'read'):
                    sim_dict = util.read_scalar(save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat")
                    print(sim_dict)

                error_array[i][j][k] = k + kl*(j + i*jl)

    # if(sys.argv[1] == 'run'):
    #     print("Data files moved to " + save_directory)

    if(sys.argv[1] == 'plot'):
        plot_3Dheatmap(alpha_array, beta_array, eta_array)

run()





# Parameter explanation:
# Alpha:    Sigma/dx
# Beta:     (ngd*dx)/Sigma
# Eta:      Rc/Sigma

# par_dict['alpha']=      0.9352
# par_dict['beta']=       9.706
# par_dict['eta']=        5.573



























#