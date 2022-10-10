from asyncore import loop
from cProfile import label
from distutils.util import execute
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erf
import sys
import subprocess

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from settings import *
import util

class SIM:

    def __init__(self):
        pardict = util.read_info(info_file_name)
        # Initialise parameters
        # pardict['N']=          500000.0
        # pardict['rh']=         0.02609300415934458
        # pardict['alpha'], pardict['beta'], pardict['eta'] = util.par_given_error(1.e-3)
        # npts = 270
        # pardict['nx']=         npts
        # pardict['ny']=         npts
        # pardict['nz']=         npts
        pardict['repeat']=     40
        pardict['prompt']=     -1
        pardict['dt']=         0.1
        pardict['Fref']=       pardict['rh']
        pardict['packrep']=    50

        self.pars = pardict.copy()
        self.reference_pars = pardict.copy()

        self.search_grid_shape = (6, 3, 6)
        self.npts_search_shape = 10


    def start_loop(self):
        nphi = 4
        nn = 10
        loopshape = (nphi, nn)
        time_compute_array = np.zeros(loopshape)
        Verror_array = np.zeros(loopshape)
        Werror_array = np.zeros(loopshape)
        phi_array = np.zeros(loopshape)
        n_array = np.zeros(loopshape)

        for i in range(nphi):
            for j in range(nn):
                phi=                        0.05 + i*0.05
                self.pars['N']=             100000.0 + 100000.0*j
                self.pars['rh']=            util.compute_rad(self.pars['N'], phi)
                self.pars['Fref']=          self.pars['rh']
                if(sys.argv[1] == 'run'):
                    self.print_siminfo(i, j)
                    util.execute_random_generator(self.pars)
                    self.get_reference()

                self.pars['alpha'], self.pars['beta'], self.pars['eta'], npts, optimal_Verror, optimal_Werror, optimal_time = self.find_optimal(1.e-3)
                self.pars['nx']=         npts
                self.pars['ny']=         npts
                self.pars['nz']=         npts

                sim_dict = util.read_scalar(self.pars)
                sim_dict["Verror"] = optimal_Verror
                sim_dict["Werror"] = optimal_Werror
                sim_dict['time_compute'] = optimal_time
                self.print_scalar(sim_dict)
                util.write_scalar(self.pars, sim_dict)

                Verror_array[i, j] = sim_dict["Verror"]
                Werror_array[i, j] = sim_dict["Werror"]
                time_compute_array[i, j] = sim_dict["time_compute"]
                phi_array[i, j] = phi
                n_array[i, j] = self.pars['N']


    def get_reference(self):
        '''Generate reference data
        '''
        self.reference_pars=                self.pars.copy()
        self.reference_pars['alpha'], self.reference_pars['beta'], self.reference_pars['eta'], npts = util.par_reference(self.reference_pars['rh'])
        self.reference_pars['nx']=          npts
        self.reference_pars['ny']=          npts
        self.reference_pars['nz']=          npts

        if(sys.argv[1] == 'run'):
            save_info_name, save_scalar_name, save_data_name = util.execute(self.reference_pars, 3)

        # print("\nFinished generating reference")



    def print_scalar(self, sim_dict):
        print("(", str(self.pars['nx']), str(self.pars['alpha']), self.pars['beta'], self.pars['eta'], ") "
                                "Verror=", str(sim_dict["Verror"]),\
                                "Werror=", str(sim_dict["Werror"]),\
                                "time_compute=", str(sim_dict["time_compute"]))

    def print_siminfo(self, i, j):
        print('\n===========================================')
        print('(', i, ',', j, ') N=', str(self.pars['N']), 'rh=', str(self.pars['rh']))
        print('===========================================')


    def find_optimal(self, tol):
        time_compute_array = np.zeros(self.search_grid_shape)
        Verror_array = np.zeros(self.search_grid_shape)
        Werror_array = np.zeros(self.search_grid_shape)
        alpha_array = np.zeros(self.search_grid_shape)
        beta_array = np.zeros(self.search_grid_shape)
        eta_array = np.zeros(self.search_grid_shape)

        npts_array = np.zeros(self.npts_search_shape)
        time_compute_array_1D = np.zeros(self.npts_search_shape)
        Verror_array_1D = np.zeros(self.npts_search_shape)
        Werror_array_1D = np.zeros(self.npts_search_shape)

        # Begin custom massive loop
        for i in range(self.search_grid_shape[0]):
            for j in range(self.search_grid_shape[1]):
                for k in range(self.search_grid_shape[2]):
                    self.pars['alpha']=      0.9 + 0.05*i
                    ngd = 8.001 + j
                    self.pars['beta']=       ngd / self.pars['alpha']
                    self.pars['eta']=        4.0 + 0.4*k
                    npts = util.compute_fastfcm_npts(self.pars['rh'])
                    self.pars['nx']=         npts
                    self.pars['ny']=         npts
                    self.pars['nz']=         npts

                    if(sys.argv[1] == 'run'):
                        save_info_name, save_scalar_name, save_data_name = util.execute(self.pars)

                    sim_dict = util.read_scalar(self.pars)
                    if (sim_dict["Verror"] == -1 and sys.argv[1] == 'run'):
                        sim_dict["Verror"], sim_dict["Werror"] = self.compute_error()

                    time_compute_array[i, j, k] = sim_dict["time_compute"]
                    Verror_array[i, j, k] = sim_dict["Verror"]
                    Werror_array[i, j, k] = sim_dict["Werror"]
                    alpha_array[i, j, k] = self.pars['alpha']
                    beta_array[i, j, k] = self.pars['beta']
                    eta_array[i, j, k] = self.pars['eta']

                    self.print_scalar(sim_dict)
                    util.write_scalar(self.pars, sim_dict)

        layer = util.layer_array(Verror_array, tol)
        time_compute_layer = time_compute_array*layer
        min_time = time_compute_layer[time_compute_layer > 0].min()
        min_index = np.where(time_compute_layer == min_time)
        optimal_Verror = Verror_array[min_index][0]
        optimal_Werror = Werror_array[min_index][0]
        optimal_alpha = alpha_array[min_index][0]
        optimal_beta = beta_array[min_index][0]
        optimal_eta = eta_array[min_index][0]

        for npt in range(self.npts_search_shape):
            self.pars['alpha']=         optimal_alpha
            self.pars['beta']=          optimal_beta
            self.pars['eta']=           optimal_eta
            npts = util.compute_fastfcm_npts(self.pars['rh']) + 4*npt
            self.pars['nx']=         npts
            self.pars['ny']=         npts
            self.pars['nz']=         npts

            if(sys.argv[1] == 'run'):
                save_info_name, save_scalar_name, save_data_name = util.execute(self.pars)

            sim_dict = util.read_scalar(self.pars)
            if (sim_dict["Verror"] == -1 and sys.argv[1] == 'run'):
                sim_dict["Verror"], sim_dict["Werror"] = self.compute_error()

            time_compute_array_1D[npt] = sim_dict["time_compute"]
            Verror_array_1D[npt] = sim_dict["Verror"]
            Werror_array_1D[npt] = sim_dict["Werror"]
            npts_array[npt] = self.pars['nx']

            self.print_scalar(sim_dict)
            util.write_scalar(self.pars, sim_dict)
        
        layer = util.layer_array_1D(Verror_array_1D, tol)
        time_compute_layer_1D = time_compute_array_1D*layer
        min_time = time_compute_layer_1D[time_compute_layer_1D > 0].min()
        min_index = np.where(time_compute_layer_1D == min_time)
        optimal_npts = npts_array[min_index][0]
        optimal_Verror = Verror_array_1D[min_index][0]
        optimal_Werror = Werror_array_1D[min_index][0]

        print("Min compute time for error=" + str(tol) + " is " + str(min_time))
        print("Corresponding (alpha beta eta):" + str(optimal_alpha) + ' '\
                                                + str(optimal_beta) + ' '\
                                                + str(optimal_eta) + ' '\
                                                + str(optimal_npts) + ' ')
                        
        return optimal_alpha, optimal_beta, optimal_eta, optimal_npts, optimal_Verror, optimal_Werror, min_time


    def compute_error(self):
        vx, vy, vz, wx, wy, wz = util.read_data(cufcm_dir + 'data/simulation/simulation_data.dat', 0, int(self.pars['N']))
        vx_ref, vy_ref, vz_ref, wx_ref, wy_ref, wz_ref =\
                util.read_data(save_directory + 'simulation_data' + util.parser(self.reference_pars) + '.dat', 0, int(self.reference_pars['N']))

        v_array = np.array([vx, vy, vz])
        vref_array = np.array([vx_ref, vy_ref, vz_ref])
        w_array = np.array([wx, wy, wz])
        wref_array = np.array([wx_ref, wy_ref, wz_ref])
        
        return util.percentage_error_magnitude(v_array, vref_array), util.percentage_error_magnitude(w_array, wref_array)



# Parameter explanation:
# Alpha:    Sigma/dx
# Beta:     (ngd*dx)/Sigma
# Eta:      Rc/Sigma

# par_dict['alpha']=      0.9352
# par_dict['beta']=       9.706
# par_dict['eta']=        5.573




#