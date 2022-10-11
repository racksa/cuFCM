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
        pardict['repeat']=     50
        pardict['prompt']=     -1
        pardict['dt']=         0.1
        pardict['Fref']=       pardict['rh']
        pardict['packrep']=    50

        self.pars = pardict.copy()
        self.reference_pars = pardict.copy()

        self.search_grid_shape = (5, 1, 5)
        self.npts_search_shape = 10

        self.nphi = 5
        self.nn = 20
        loopshape = (self.nphi, self.nn)
        self.optimal_time_compute_array = np.zeros(loopshape)
        self.optimal_Verror_array = np.zeros(loopshape)
        self.optimal_Werror_array = np.zeros(loopshape)
        self.optimal_alpha_array = np.zeros(loopshape)
        self.optimal_beta_array = np.zeros(loopshape)
        self.optimal_eta_array = np.zeros(loopshape)
        self.optimal_npts_array = np.zeros(loopshape)
        self.phi_array = np.zeros(loopshape)
        self.n_array = np.zeros(loopshape)

        self.siminfo = True


    def start_loop(self):

        for i in range(3, self.nphi):
            for j in range(self.nn-10):
                phi=                        0.08 + i*0.08
                self.pars['N']=             100000.0 + 100000.0*j
                self.pars['rh']=            util.compute_rad(self.pars['N'], phi)
                self.pars['Fref']=          self.pars['rh']
                self.print_siminfo(i, j)
                if(sys.argv[1] == 'run'):
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

                self.optimal_time_compute_array[i, j] = sim_dict["time_compute"]
                self.optimal_Verror_array[i, j] = sim_dict["Verror"]
                self.optimal_Werror_array[i, j] = sim_dict["Werror"]
                self.optimal_alpha_array[i, j] = self.pars['alpha']
                self.optimal_beta_array[i, j] = self.pars['beta']
                self.optimal_eta_array[i, j] = self.pars['eta']
                self.optimal_npts_array[i, j] = self.pars['nx']

                self.phi_array[i, j] = phi
                self.n_array[i, j] = self.pars['N']
        
        self.save_optimal_arrays()


    def analyse(self):
        self.load_optimal_arrays()
        print('Optimal parameters:')
        optimal_parameters = [[self.optimal_alpha_array[i, j], self.optimal_beta_array[i, j],\
                self.optimal_eta_array[i, j], self.optimal_npts_array[i, j]] \
                for i in range(self.nphi) for j in range(self.nn)]
        for i in optimal_parameters:
            print(i)

        print(self.optimal_time_compute_array)

    
    def plot_optimal(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        for i, n in enumerate(self.n_array):
            ptps_array = self.n_array[i]/self.optimal_time_compute_array[i]
            ax.plot(self.n_array[i], ptps_array, marker='o', c=util.color_codex[i], label=str(self.phi_array[i][0]))
        ax.legend()
        
        # adding title and labels
        ax.set_title("PTPS vs. N")
        ax.set_xlabel('N')
        ax.set_ylabel('PTPS')
        plt.savefig('img/npts.eps', format='eps')
        plt.show()


    def save_optimal_arrays(self):
        np.save(save_directory + 'optimal_timecompute_array.npy', self.optimal_time_compute_array)
        np.save(save_directory + 'optimal_Verror_array.npy', self.optimal_Verror_array)
        np.save(save_directory + 'optimal_Werror_array.npy', self.optimal_Werror_array)
        np.save(save_directory + 'optimal_alpha_array.npy', self.optimal_alpha_array)
        np.save(save_directory + 'optimal_beta_array.npy', self.optimal_beta_array)
        np.save(save_directory + 'optimal_eta_array.npy', self.optimal_eta_array)
        np.save(save_directory + 'optimal_npts_array.npy', self.optimal_npts_array)
        np.save(save_directory + 'phi_array.npy', self.phi_array)
        np.save(save_directory + 'n_array.npy', self.n_array)


    def load_optimal_arrays(self):
        self.optimal_time_compute_array = np.load(save_directory + 'optimal_timecompute_array.npy')
        self.optimal_Verror_array = np.load(save_directory + 'optimal_Verror_array.npy')
        self.optimal_Werror_array = np.load(save_directory + 'optimal_Werror_array.npy')
        self.optimal_alpha_array = np.load(save_directory + 'optimal_alpha_array.npy')
        self.optimal_beta_array = np.load(save_directory + 'optimal_beta_array.npy')
        self.optimal_eta_array = np.load(save_directory + 'optimal_eta_array.npy')
        self.optimal_npts_array = np.load(save_directory + 'optimal_npts_array.npy')
        self.phi_array = np.load(save_directory + 'phi_array.npy')
        self.n_array = np.load(save_directory + 'n_array.npy')


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
        if(self.siminfo):
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
                    self.pars['alpha']=      0.93 + 0.01*i
                    ngd=                     9.00
                    self.pars['beta']=       ngd / self.pars['alpha']
                    self.pars['eta']=        4.7 + 0.1*k
                    npts = util.compute_fastfcm_npts(self.pars['rh'])
                    self.pars['nx']=         npts
                    self.pars['ny']=         npts
                    self.pars['nz']=         npts

                    if(sys.argv[1] == 'run'):
                        save_info_name, save_scalar_name, save_data_name = util.execute(self.pars)

                    sim_dict = util.read_scalar(self.pars)
                    if(sim_dict):
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

        self.pars['alpha']=         optimal_alpha
        self.pars['beta']=          optimal_beta
        self.pars['eta']=           optimal_eta

        for npt in range(self.npts_search_shape):
            npts = util.compute_fastfcm_npts(self.pars['rh']) + 4*(npt+1)
            self.pars['nx']=         npts
            self.pars['ny']=         npts
            self.pars['nz']=         npts

            if(sys.argv[1] == 'run'):
                save_info_name, save_scalar_name, save_data_name = util.execute(self.pars)

            sim_dict = util.read_scalar(self.pars)

            if(sim_dict):
                if (sim_dict["Verror"] == -1 and sys.argv[1] == 'run'):
                    sim_dict["Verror"], sim_dict["Werror"] = self.compute_error()

                time_compute_array_1D[npt] = sim_dict["time_compute"]
                Verror_array_1D[npt] = sim_dict["Verror"]
                Werror_array_1D[npt] = sim_dict["Werror"]
                npts_array[npt] = self.pars['nx']

                self.print_scalar(sim_dict)
                util.write_scalar(self.pars, sim_dict)
        
        layer_1D = util.layer_array_1D(Verror_array_1D, tol)
        time_compute_layer_1D = time_compute_array_1D*layer_1D
        min_time = time_compute_layer_1D[time_compute_layer_1D > 0].min()
        min_index = np.where(time_compute_layer_1D == min_time)
        optimal_Verror = Verror_array_1D[min_index][0]
        optimal_Werror = Werror_array_1D[min_index][0]
        optimal_npts = npts_array[min_index][0]


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