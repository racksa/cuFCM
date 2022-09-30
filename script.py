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
    save_directory = "data/simulation/20220929_fcm_error1/"

    l = 1
    il, jl, kl = l, l, l
    time_compute_array = np.zeros((l, l, l))
    Verror_array = np.zeros((l, l, l))
    Werror_array = np.zeros((l, l, l))
    alpha_array = np.zeros((l, l, l))
    beta_array = np.zeros((l, l, l))
    eta_array = np.zeros((l, l, l))

    nl = 1
    npts_array = np.zeros(nl)
    time_compute_npts_array = np.zeros(nl)
    Verror_npts_array = np.zeros(nl)
    Werror_npts_array = np.zeros(nl)

    tol_list = np.array([5.e-3, 1.e-3, 5.e-4, 1.e-4, 5.e-5, 1.e-5])

    par_dict = util.read_info(info_file_name)
    # Begin custom massive loop
    for i in range(1):
        for j in range(1):
            for k in range(1):
                for npt in range(nl):

                    par_dict['N']=          500000.0
                    par_dict['rh']=         0.02609300415934458
                    par_dict['alpha'], par_dict['beta'], par_dict['eta'] = util.par_given_error(1.e-3)
                    # par_dict['alpha'], par_dict['beta'], par_dict['eta'] = util.fcm_par_given_error(1.e-3, par_dict['rh'])
                    npts = 270
                    par_dict['nx']=         npts
                    par_dict['ny']=         npts
                    par_dict['nz']=         npts
                    par_dict['repeat']=     100
                    par_dict['prompt']=     10

                    alpha_array[i][j][k] = par_dict['alpha']
                    beta_array[i][j][k] = par_dict['beta']
                    eta_array[i][j][k] = par_dict['eta']

                    npts_array[npt] = npts

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

                    if(sys.argv[1] == 'plot3' or sys.argv[1] == 'plot1' or sys.argv[1] == 'plot_npts'):
                        sim_dict = util.read_scalar(save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat")
                        if(sys.argv[1] == 'plot3' or sys.argv[1] == 'plot1'):
                            time_compute_array[i][j][k] = sim_dict['time_compute']
                            Verror_array[i][j][k] = sim_dict['Verror']
                            Werror_array[i][j][k] = sim_dict['Werror']
                        if(sys.argv[1] == 'plot_npts'):
                            time_compute_npts_array[npt] = sim_dict['time_compute']
                            Verror_npts_array[npt] = sim_dict['Verror']
                            Werror_npts_array[npt] = sim_dict['Werror']

                    if(sys.argv[1] == 'clean'):
                        subprocess.call("rm -f " + save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat", shell=True)
                        subprocess.call("rm -f " + save_directory + "simulation_info" + util.parser(par_dict) + ".dat", shell=True)

    if(sys.argv[1] == 'plot1'):
        option = "1n"
        if(len(sys.argv)>2):
            option = sys.argv[2]
        util.plot_1D_fit(alpha_array, beta_array, eta_array, Verror_array, option)

    if(sys.argv[1] == 'plot3'):
        option = [1, 1]
        if(len(sys.argv)>2):
            option[0] = float(sys.argv[2])
        if(len(sys.argv)>3):
            option[1] = float(sys.argv[3])
        util.plot_3Dheatmap(alpha_array, beta_array, eta_array, Verror_array, time_compute_array, option)

    if(sys.argv[1] == 'plot_npts'):
        util.plot_npts(npts_array, Verror_npts_array, time_compute_npts_array)

run()


# Parameter explanation:
# Alpha:    Sigma/dx
# Beta:     (ngd*dx)/Sigma
# Eta:      Rc/Sigma

# par_dict['alpha']=      0.9352
# par_dict['beta']=       9.706
# par_dict['eta']=        5.573




#