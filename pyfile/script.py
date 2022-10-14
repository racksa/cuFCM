from distutils.log import error
import sys
from pathlib import Path
import os
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pylab import *
from settings import *
import util
import simulator

sim = simulator.SIM()

if(sys.argv[1] == 'run' or sys.argv[1] == 'read'):
    sim.start_loop()

if(sys.argv[1] == 'test'):
    sim.run_test()

if(sys.argv[1] == 'read_np'):
    sim.analyse()
    sim.plot_optimal()


# def run():

#     l = 1
#     il, jl, kl = l, l, l
#     time_compute_array = np.zeros((l, l, l))
#     Verror_array = np.zeros((l, l, l))
#     Werror_array = np.zeros((l, l, l))
#     alpha_array = np.zeros((l, l, l))
#     beta_array = np.zeros((l, l, l))
#     eta_array = np.zeros((l, l, l))

#     nl = 1
#     svar_array = np.zeros(nl)
#     time_compute_svar_array = np.zeros(nl)
#     time_compute_svar_array2 = np.zeros(nl)
#     Verror_svar_array = np.zeros(nl)
#     Werror_svar_array = np.zeros(nl)
    

#     tol_list = np.array([5.e-3, 1.e-3, 5.e-4, 1.e-4, 5.e-5, 1.e-5])

#     par_dict = util.read_info(info_file_name)
    
#     solver_type = 1
#     # 0: FCM
#     # 1: Fast FCM

#     if(sys.argv[1] == 'plot_n'):
#         solver_start = 0
#         solver_end = 2
#     else:
#         solver_start = solver_type
#         solver_end = solver_type + 1

#     # Begin custom massive loop
#     for i in range(1):
#         for j in range(1):
#             for k in range(1):
#                 for svar in range(nl):
#                     for current_solver in range(solver_start, solver_end):
                    
#                         n = 500000 + 100000*svar
#                         par_dict['N']=          n
#                         par_dict['rh']=         0.02609300415934458
#                         # par_dict['rh']=         compute_rad(par_dict['N'], 0.15)
#                         if(current_solver == 1):
#                             par_dict['alpha'], par_dict['beta'], par_dict['eta'] = util.par_given_error(1.e-3)
#                             npts = util.compute_fastfcm_npts(par_dict['rh'])
#                         elif(current_solver == 0):
#                             par_dict['alpha'], par_dict['beta'], par_dict['eta'] = util.fcm_par_given_error(1.e-3, par_dict['rh'])
#                             dx = par_dict['rh']/(par_dict['alpha'] * np.sqrt(np.pi))
#                             npts = 2*int(np.pi/dx)
#                             dx = 2*np.pi/npts
#                             par_dict['alpha'] = par_dict['rh']/(dx * np.sqrt(np.pi))
#                             par_dict['eta'] = (np.pi/12)/(par_dict['alpha'] * dx)
#                         par_dict['nx']=         npts
#                         par_dict['ny']=         npts
#                         par_dict['nz']=         npts
#                         par_dict['repeat']=     40
#                         par_dict['prompt']=     10
#                         par_dict['dt']=         0.1
#                         par_dict['Fref']=       par_dict['rh']
#                         par_dict['packrep']=    50

#                         alpha_array[i][j][k] = par_dict['alpha']
#                         beta_array[i][j][k] = par_dict['beta']
#                         eta_array[i][j][k] = par_dict['eta']

#                         # single_var_array[svar] = npts
#                         svar_array[svar] = n

#                         if(sys.argv[1] == 'run'):
#                             for key in par_dict:
#                                 util.replace(key, str(par_dict[key]), info_file_name)

#                             subprocess.call("./bin/CUFCM", shell=True)
#                             if(current_solver == 0):
#                                 save_info_name, save_scalar_name, save_data_name = util.savefile(par_dict, save_directory, 2)
#                             if(current_solver == 1):
#                                 save_info_name, save_scalar_name, save_data_name = util.savefile(par_dict, save_directory2, 2)

#                             sim_dict = util.read_scalar(save_scalar_name)
#                             print("(", str(npts), str(par_dict['alpha']), par_dict['beta'], par_dict['eta'], ") "
#                                 "Verror=", str(sim_dict["Verror"]),\
#                                 "Werror=", str(sim_dict["Werror"]),\
#                                 "time_compute=", str(sim_dict["time_compute"]))

#                         if(sys.argv[1] == 'plot3' or sys.argv[1] == 'plot1' or sys.argv[1] == 'plot_npts'):
#                             sim_dict = util.read_scalar(save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat")
#                             if(sys.argv[1] == 'plot3' or sys.argv[1] == 'plot1'):
#                                 time_compute_array[i][j][k] = sim_dict['time_compute']
#                                 Verror_array[i][j][k] = sim_dict['Verror']
#                                 Werror_array[i][j][k] = sim_dict['Werror']
#                             if(sys.argv[1] == 'plot_npts'):
#                                 time_compute_svar_array[svar] = sim_dict['time_compute']
#                                 Verror_svar_array[svar] = sim_dict['Verror']
#                                 Werror_svar_array[svar] = sim_dict['Werror']
#                         if(sys.argv[1] == 'plot_n'):
#                             if(current_solver == 0):
#                                 sim_dict = util.read_scalar(save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat")
#                                 time_compute_svar_array[svar] = sim_dict['time_compute']
#                             elif(current_solver == 1):
#                                 sim_dict2 = util.read_scalar(save_directory2 + "simulation_scalar" + util.parser(par_dict) + ".dat")
#                                 time_compute_svar_array2[svar] = sim_dict2['time_compute']

#                         if(sys.argv[1] == 'clean'):
#                             subprocess.call("rm -f " + save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat", shell=True)
#                             subprocess.call("rm -f " + save_directory + "simulation_info" + util.parser(par_dict) + ".dat", shell=True)

#     if(sys.argv[1] == 'plot1'):
#         option = "1n"
#         if(len(sys.argv)>2):
#             option = sys.argv[2]
#         util.plot_1D_fit(alpha_array, beta_array, eta_array, Verror_array, option)

#     if(sys.argv[1] == 'plot3'):
#         option = [1, 1]
#         if(len(sys.argv)>2):
#             option[0] = float(sys.argv[2])
#         if(len(sys.argv)>3):
#             option[1] = float(sys.argv[3])
#         util.plot_3Dheatmap(alpha_array, beta_array, eta_array, Verror_array, time_compute_array, option)

#     if(sys.argv[1] == 'plot_npts'):
#         util.plot_npts(svar_array, Verror_svar_array, time_compute_svar_array)

#     if(sys.argv[1] == 'plot_n'):
#         util.plot_n(svar_array, time_compute_svar_array, time_compute_svar_array2)

# run()





# Parameter explanation:
# Alpha:    Sigma/dx
# Beta:     (ngd*dx)/Sigma
# Eta:      Rc/Sigma

# par_dict['alpha']=      0.9352
# par_dict['beta']=       9.706
# par_dict['eta']=        5.573




#