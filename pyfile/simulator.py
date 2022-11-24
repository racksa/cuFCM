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
        pardict, filedict = util.read_info(info_file_name)
        # Initialise parameters
        pardict['repeat']=     40
        pardict['prompt']=     -1
        pardict['dt']=         0.1
        pardict['Fref']=       pardict['rh']
        pardict['packrep']=    50
        pardict['boxsize']=    np.pi*2
        pardict['Ffac']=       1.0
        pardict['Tfac']=       1.0

        self.pars = pardict.copy()
        self.datafiles = filedict.copy()
        self.reference_pars = pardict.copy()

        self.search_grid_shape = (1, 1, 1, 20+1) # alpha, beta, eta, npts

        self.nphi = 10
        self.nn = 12
        loopshape = (self.nphi, self.nn)
        self.optimal_time_compute_array = np.zeros(loopshape)
        self.optimal_Verror_array = np.zeros(loopshape)
        self.optimal_Werror_array = np.zeros(loopshape)
        self.optimal_alpha_array = np.zeros(loopshape)
        self.optimal_beta_array = np.zeros(loopshape)
        self.optimal_eta_array = np.zeros(loopshape)
        self.optimal_npts_array = np.zeros(loopshape)
        self.phi_array = np.zeros(loopshape)
        self.rh_array = np.zeros(loopshape)
        self.n_array = np.zeros(loopshape)
        self.sigfac_array = np.zeros(loopshape)

        self.siminfo = True


    def start_loop(self):
        self.datafiles['$posfile'] = './data/init_data/new/pos_data.dat'
        self.datafiles['$forcefile'] = './data/init_data/new/force_data.dat'
        self.datafiles['$torquefile'] = './data/init_data/new/torque_data.dat'
        self.pars['checkerror'] = 0

        for i in range(self.nphi):
            for j in range(self.nn):
                phi=                        0.0001 * 2**j
                self.pars['rh']=            0.024 + 0.006*i
                self.pars['N']=             util.compute_N(phi, self.pars['rh'])

                # self.pars['N']=             int(1000*2**j)
                # self.pars['rh']=            0.024 * 2**i
                phi=                        util.compute_phi(self.pars['N'], self.pars['rh'])
                self.pars['Fref']=          self.pars['rh']

                self.phi_array[i, j] = phi
                self.rh_array[i, j] = self.pars['rh']
                self.n_array[i, j] = self.pars['N']

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
                self.sigfac_array[i, j] = (self.pars['alpha']*self.pars['boxsize']/self.pars['nx'])/(self.pars['rh']/np.sqrt(np.pi))

        
        self.save_optimal_arrays()

    def get_reference(self):
        '''Generate reference data
        '''
        self.reference_pars=                self.pars.copy()
        self.reference_pars['alpha']        = 1.5
        self.reference_pars['beta']         = 15.0 / self.reference_pars['alpha']
        self.reference_pars['eta']          = round(8.0, 1)
        npts                                = min( int(0.026/self.reference_pars['rh'] * 270)/2 * 2, 460)
        self.reference_pars['nx']=          npts
        self.reference_pars['ny']=          npts
        self.reference_pars['nz']=          npts
        self.reference_pars['repeat']=      1
        self.reference_pars['prompt']=      -1

        if(sys.argv[1] == 'run' or sys.argv[1] == 'test'):
            util.execute([self.reference_pars, self.datafiles], solver=1, mode=3)

        print("Finished generating reference")


    def find_optimal(self, tol):
        time_compute_array = np.zeros(self.search_grid_shape)
        Verror_array = np.zeros(self.search_grid_shape)
        Werror_array = np.zeros(self.search_grid_shape)
        alpha_array = np.zeros(self.search_grid_shape)
        beta_array = np.zeros(self.search_grid_shape)
        eta_array = np.zeros(self.search_grid_shape)
        npts_array = np.zeros(self.search_grid_shape)

        # Begin custom massive loop
        for l in range(self.search_grid_shape[3]):
            for i in range(self.search_grid_shape[0]):
                for j in range(self.search_grid_shape[1]):
                    for k in range(self.search_grid_shape[2]):
                        self.pars['alpha']=      1.0 + 0.02*i
                        self.pars['beta']=       (9. + j ) / self.pars['alpha']
                        self.pars['eta']=        5.0 + np.exp(-8e-6*self.pars['N'])

                        if(HIsolver==1):
                            npts = min(100 + 18*l, int(self.pars['boxsize']/(self.pars['rh']/np.sqrt(np.pi))))
                        if(HIsolver==0):
                            npts = 200 + 18*l
                        self.pars['nx']=         npts
                        self.pars['ny']=         npts
                        self.pars['nz']=         npts

                        if HIsolver == 0:
                            dx = self.pars['boxsize']/npts
                            self.pars['alpha']= (self.pars['rh']/np.sqrt(np.pi))/dx
                            self.pars['eta']= self.pars['eta']/self.pars['alpha']

                        if(sys.argv[1] == 'run'):
                            util.execute([self.pars, self.datafiles], solver=HIsolver, mode=2)

                        sim_dict = util.read_scalar(self.pars)
                        if(sim_dict):
                            if (sim_dict["Verror"] == -1 and sys.argv[1] == 'run'):
                                sim_dict["Verror"], sim_dict["Werror"] = self.compute_error()

                            time_compute_array[i, j, k, l] = sim_dict["time_compute"]
                            Verror_array[i, j, k, l] = sim_dict["Verror"]
                            Werror_array[i, j, k, l] = sim_dict["Werror"]
                            alpha_array[i, j, k, l] = self.pars['alpha']
                            beta_array[i, j, k, l] = self.pars['beta']
                            eta_array[i, j, k, l] = self.pars['eta']
                            npts_array[i, j, k, l] = self.pars['nx']

                            self.print_scalar(sim_dict)
                            util.write_scalar(self.pars, sim_dict)

        my_filter = util.filter_array(Verror_array, tol)
        time_compute_layer = time_compute_array*my_filter
        if(not len(time_compute_layer[time_compute_layer > 0]) == 0):
            # Tolerance reached by at least one value
            min_time = time_compute_layer[time_compute_layer > 0].min()
            min_index = np.where(time_compute_layer == min_time)
        else:
            # Tolerance not reached
            # min_time = time_compute_array[time_compute_array > 0].min()
            min_error = Verror_array.min()
            min_index = np.where(Verror_array == min_error)
            min_time = time_compute_array[min_index][0]
            print('Tolerance not satisfied in optimal finding. Carry on with the smallest error.')

        optimal_Verror = Verror_array[min_index][0]
        optimal_Werror = Werror_array[min_index][0]
        optimal_alpha = alpha_array[min_index][0]
        optimal_beta = beta_array[min_index][0]
        optimal_eta = eta_array[min_index][0]
        optimal_npts = npts_array[min_index][0]


        print("Min compute time for error=" + str(tol) + " is " + str(min_time))
        print("Corresponding (alpha beta eta npts):" + str(optimal_alpha) + ' '\
                                                + str(optimal_beta) + ' '\
                                                + str(optimal_eta) + ' '\
                                                + str(optimal_npts) + ' ')
                        
        return optimal_alpha, optimal_beta, optimal_eta, optimal_npts, optimal_Verror, optimal_Werror, min_time


    def run_test(self):
        self.datafiles['$posfile'] = './data/init_data/N500000/pos-N500000-rh02609300-2.dat'
        self.datafiles['$forcefile'] = './data/init_data/N500000/force-N500000-rh02609300.dat'
        self.datafiles['$torquefile'] = './data/init_data/N500000/force-N500000-rh02609300-2.dat'
        self.pars['checkerror'] = 1

        fac = 1.0
        self.pars['N']=          500000
        self.pars['rh']=         0.02609300415934458*fac
        self.pars['alpha']=      1.0
        self.pars['beta']=       9.0
        self.pars['eta']=        4.8
        npts = 320  # Fast FCM
        # npts = 480  # Regular FCM

        self.pars['nx']=         npts
        self.pars['ny']=         npts
        self.pars['nz']=         npts
        self.pars['Fref']=       1.0
        self.pars['repeat']=     40
        self.pars['prompt']=     10
        self.pars['boxsize']=    np.pi*2*fac
        self.pars['Ffac']=       fac**2
        self.pars['Tfac']=       fac**3

        if HIsolver == 0:
            dx = self.pars['boxsize']/npts
            self.pars['alpha']= (self.pars['rh']/np.sqrt(np.pi))/dx
            self.pars['eta']= self.pars['eta']/self.pars['alpha']

        util.execute([self.pars, self.datafiles], solver=HIsolver, mode=3)

        self.plot_pie_chart_of_time()


    def print_scalar(self, sim_dict):
        if(self.siminfo):
            print("(", str(self.pars['nx']), str(self.pars['alpha']), self.pars['beta'], self.pars['eta'], ") "
                                    "Verror=", str(sim_dict["Verror"]),\
                                    "Werror=", str(sim_dict["Werror"]),\
                                    "time_compute=", str(sim_dict["time_compute"]),\
                                    "PTPS=", str(self.pars['N']/sim_dict["time_compute"]),\
                                    "Sig/sig=", str((self.pars['alpha']*self.pars['boxsize']/self.pars['nx'])/(self.pars['rh']/np.sqrt(np.pi)))
                 )

    def print_siminfo(self, i, j):
        print('\n===========================================')
        print('(', i, ',', j, ') N=', str(self.pars['N']), 'rh=', str(self.pars['rh']), 'phi=', str(self.phi_array[i, j]), 'solver=', str(HIsolver))
        print('===========================================')


    def compute_error(self):
        vx, vy, vz, wx, wy, wz = util.read_data(cufcm_dir + 'data/simulation/simulation_data.dat', 0, int(self.pars['N']))
        vx_ref, vy_ref, vz_ref, wx_ref, wy_ref, wz_ref =\
                util.read_data(fastfcm_directory + 'simulation_data' + util.parser(self.reference_pars) + '.dat', 0, int(self.reference_pars['N']))

        v_array = np.array([vx, vy, vz])
        vref_array = np.array([vx_ref, vy_ref, vz_ref])
        w_array = np.array([wx, wy, wz])
        wref_array = np.array([wx_ref, wy_ref, wz_ref])
        
        return util.percentage_error_magnitude(v_array, vref_array), util.percentage_error_magnitude(w_array, wref_array)


    def analyse(self):
        self.load_optimal_arrays()
        print('Optimal parameters:')
        optimal_parameters = [[i, j, self.optimal_alpha_array[i, j], self.optimal_beta_array[i, j],\
                self.optimal_eta_array[i, j], self.optimal_npts_array[i, j]] \
                for i in range(self.nphi) for j in range(self.nn)]
        optimal_times = [self.optimal_time_compute_array[i, j] \
                for i in range(self.nphi) for j in range(self.nn)]
        for i, pars in enumerate(optimal_parameters):
            print(pars, 'PTPS=', self.pars['N']/optimal_times[i])
            if ((i+1)%self.nn == 0):
                print('--------------------')

        print(self.optimal_time_compute_array)

    def analyse_and_plot_both(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        for j in range(2):
            self.mod_solver(j)
            self.analyse()

            for i, n in enumerate(self.n_array):
                if (HIsolver==0):
                    label = 'FCM a=' + str(self.rh_array[i][0])
                    marker = 'o'
                    linestyle='solid'
                if (HIsolver==1):
                    label = 'Fast FCM a=' + str(self.rh_array[i][0])
                    marker = 's'
                    linestyle='dashed'

                ptps_array = self.n_array[i]/self.optimal_time_compute_array[i]
                ax.plot(self.phi_array[i], ptps_array, marker=marker, linestyle=linestyle, c=util.color_codex[i], label=label)
        
        # adding title and labels
        ax.legend()
        ax.set_title(r"PTPS vs. $\phi$")
        ax.set_xlabel(r'$phi$')
        ax.set_ylabel('PTPS')
        ax.set_xscale('log')
        plt.savefig('img/ptps_combined.eps', format='eps')
        plt.show()

    
    def plot_ptps(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        for i, n in enumerate(self.n_array):
            ptps_array = self.n_array[i]/self.optimal_time_compute_array[i]
            ax.plot(self.n_array[i], ptps_array, marker='o', c=util.color_codex[i], label=r'$a$=' + str(round(self.rh_array[i][0], 4)))
        ax.legend()
        
        # adding title and labels
        ax.set_title("PTPS vs. N")
        ax.set_xlabel('N')
        ax.set_ylabel('PTPS')
        ax.set_xscale('log')
        plt.savefig('img/ptps_optimal.eps', format='eps')
        plt.show()

    def plot_ptps_vs_phi(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        for i, n in enumerate(self.phi_array):
            ptps_array = self.n_array[i]/self.optimal_time_compute_array[i]
            ax.plot(self.phi_array[i], ptps_array, marker='o', c=util.color_codex[i], label=r'$a$=' + str(round(self.rh_array[i][0], 4)))
        ax.legend()
        
        # adding title and labels
        ax.set_title(r"PTPS vs. $\phi$")
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel('PTPS')
        ax.set_xscale('log')
        plt.savefig('img/ptps_vs_phi_optimal.eps', format='eps')
        plt.show()

    
    def plot_sigfac(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        for i, n in enumerate(self.n_array):
            print(self.sigfac_array[i])
            ax.plot(self.n_array[i], self.sigfac_array[i], marker='o', c=util.color_codex[i], label=r'$a$=' + str(round(self.rh_array[i][0],4)))
        ax.legend()
        
        # adding title and labels
        ax.set_title(r"$\Sigma/\sigma$ vs. N")
        ax.set_xlabel('N')
        ax.set_ylabel(r'$\Sigma/\sigma$')
        ax.set_xscale('log')
        plt.savefig('img/sigfac_optimal.eps', format='eps')
        plt.show()

    def plot_pie_chart_of_time(self):
        sim_dict = util.read_scalar(self.pars)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        keys = ['time_hashing', 'time_spreading', 'time_FFT', 'time_gathering', 'time_correction']
        labels = ['Hashing', 'Spreading', 'FFT + flow solving', 'Gathering', 'Correction']
        time_compute = sim_dict['time_compute']
        sizes = [sim_dict[key]/time_compute for key in keys]
        explode = (0, 0, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        ax.set_title("Time of Fast FCM")
        ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.savefig('img/pie_chart_of_time.eps', format='eps')
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
        np.save(save_directory + 'rh_array.npy', self.rh_array)
        np.save(save_directory + 'n_array.npy', self.n_array)
        np.save(save_directory + 'sigfac_array.npy', self.sigfac_array)


    def load_optimal_arrays(self):
        self.optimal_time_compute_array = np.load(save_directory + 'optimal_timecompute_array.npy')
        self.optimal_Verror_array = np.load(save_directory + 'optimal_Verror_array.npy')
        self.optimal_Werror_array = np.load(save_directory + 'optimal_Werror_array.npy')
        self.optimal_alpha_array = np.load(save_directory + 'optimal_alpha_array.npy')
        self.optimal_beta_array = np.load(save_directory + 'optimal_beta_array.npy')
        self.optimal_eta_array = np.load(save_directory + 'optimal_eta_array.npy')
        self.optimal_npts_array = np.load(save_directory + 'optimal_npts_array.npy')
        self.phi_array = np.load(save_directory + 'phi_array.npy')
        self.rh_array = np.load(save_directory + 'rh_array.npy')
        self.n_array = np.load(save_directory + 'n_array.npy')
        self.sigfac_array = np.load(save_directory + 'sigfac_array.npy')

    def mod_solver(self, n):
        global HIsolver
        global save_directory
        HIsolver = n
        if (HIsolver==0):
            save_directory = fcm_directory
        if (HIsolver==1):
            save_directory = fastfcm_directory
            
# Parameter explanation:
# Alpha:    Sigma/dx
# Beta:     (ngd*dx)/Sigma
# Eta:      Rc/Sigma

# par_dict['alpha']=      0.9352
# par_dict['beta']=       9.706
# par_dict['eta']=        5.573




#