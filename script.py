from distutils.log import error
import sys
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pylab import *
from pyfile import util
from pyfile.util import alpha_expr, beta_expr, eta_expr

def run():
    info_file_name = "simulation_info"
    save_directory = "data/simulation/20220926_error1/"

    l = 8
    il, jl, kl = l, l, l
    time_compute_array = np.zeros((l, l, l))
    Verror_array = np.zeros((l, l, l))
    Werror_array = np.zeros((l, l, l))
    alpha_array = np.zeros((l, l, l))
    beta_array = np.zeros((l, l, l))
    eta_array = np.zeros((l, l, l))

    par_dict = util.read_info(info_file_name)
    # Begin custom massive loop
    for i in range(il):
        for j in range(jl):
            for k in range(kl):

                tol = 0.0005
                npts = 256
                par_dict['N']=          500000.0
                par_dict['rh']=         0.02609300415934458
                par_dict['alpha']=      0.8 + 0.07*i
                par_dict['beta']=       9.0 + 0.6*j
                par_dict['eta']=        5.5 + 0.35*k
                # par_dict['alpha']=      alpha_expr(tol)
                # par_dict['beta']=       beta_expr(tol)
                # par_dict['eta']=        eta_expr(tol)
                par_dict['nx']=         npts
                par_dict['ny']=         npts
                par_dict['nz']=         npts
                par_dict['repeat']=     1
                par_dict['prompt']=     -1

                alpha_array[i][j][k] = par_dict['alpha']
                beta_array[i][j][k] = par_dict['beta']
                eta_array[i][j][k] = par_dict['eta']

                if(sys.argv[1] == 'run'):
                    for key in par_dict:
                        util.replace(key, str(par_dict[key]), info_file_name)

                    subprocess.call("./bin/CUFCM.o", shell=True)
                    save_info_name, save_scalar_name, save_data_name = util.savefile(par_dict, save_directory, 2)

                    sim_dict = util.read_scalar(save_scalar_name)
                    print("(", str(npts), str(par_dict['alpha']), par_dict['beta'], par_dict['eta'], ") "
                          "Verror=", str(sim_dict["Verror"]),\
                          "Werror=", str(sim_dict["Werror"]),\
                          "time_compute=", str(sim_dict["time_compute"]))

                if(sys.argv[1] == 'plot3' or sys.argv[1] == 'plot1'):
                    sim_dict = util.read_scalar(save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat")
                    time_compute_array[i][j][k] = sim_dict['time_compute']
                    Verror_array[i][j][k] = sim_dict['Verror']
                    Werror_array[i][j][k] = sim_dict['Werror']

                if(sys.argv[1] == 'clean'):
                    subprocess.call("rm -f " + save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat", shell=True)
                    subprocess.call("rm -f " + save_directory + "simulation_info" + util.parser(par_dict) + ".dat", shell=True)

    # if(sys.argv[1] == 'run'):
    #     print("Data files moved to " + save_directory)

    if(sys.argv[1] == 'plot1'):
        option = "1n"
        if(len(sys.argv)>1):
            option = sys.argv[2]
        util.plot_1D_fit(alpha_array, beta_array, eta_array, Verror_array, option)

    if(sys.argv[1] == 'plot3'):
        util.plot_3Dheatmap(alpha_array, beta_array, eta_array, Verror_array)

run()


# Parameter explanation:
# Alpha:    Sigma/dx
# Beta:     (ngd*dx)/Sigma
# Eta:      Rc/Sigma

# par_dict['alpha']=      0.9352
# par_dict['beta']=       9.706
# par_dict['eta']=        5.573




#