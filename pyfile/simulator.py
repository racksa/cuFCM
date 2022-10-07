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
        pardict['N']=          100000.0
        pardict['rh']=         0.02609300415934458
        pardict['alpha'], pardict['beta'], pardict['eta'] = util.par_given_error(1.e-3)
        npts = 270
        pardict['nx']=         npts
        pardict['ny']=         npts
        pardict['nz']=         npts
        pardict['repeat']=     10
        pardict['prompt']=     -1
        pardict['dt']=         0.1
        pardict['Fref']=       pardict['rh']
        pardict['packrep']=    50

        self.pars = pardict.copy()
        self.reference_pars = pardict.copy()



    def start_loop(self, solver):
        nphi = 1
        nn = 1
        loopshape = (nphi, nn)
        time_compute_array = np.zeros(loopshape)
        Verror_array = np.zeros(loopshape)
        Werror_array = np.zeros(loopshape)
        phi_array = np.zeros(loopshape)
        n_array = np.zeros(loopshape)

        for i in range(nphi):
            for j in range(nn):
                phi=                        0.05 + i*0.05
                self.pars['N']=             100000 + 10*j
                self.pars['rh']=            util.compute_rad(self.pars['N'], phi)
                if(sys.argv[1] == 'run'):
                    util.execute(self.pars, -1)
                    self.get_reference(solver)

                self.pars['alpha'], self.pars['beta'], self.pars['eta'], npts = self.find_optimal(2, 1.e-3, solver)
                self.pars['nx']=         npts
                self.pars['ny']=         npts
                self.pars['nz']=         npts

                sim_dict = util.read_scalar(self.pars, solver)
                self.print_scalar(sim_dict)

                Verror_array[i, j] = sim_dict["Verror"]
                Werror_array[i, j] = sim_dict["Werror"]
                time_compute_array[i, j] = sim_dict["time_compute"]
                phi_array[i, j] = phi
                n_array[i, j] = self.pars['N']



    def get_reference(self, solver):
        '''Generate reference data
        '''
        self.reference_pars=                self.pars.copy()
        self.reference_pars['alpha'], self.reference_pars['beta'], self.reference_pars['eta'], npts = util.par_reference(self.reference_pars['rh'])
        self.reference_pars['nx']=          npts
        self.reference_pars['ny']=          npts
        self.reference_pars['nz']=          npts

        if(sys.argv[1] == 'run'):
            save_info_name, save_scalar_name, save_data_name = util.execute(self.reference_pars, solver, 3)

        print("\nFinished generating reference")



    def print_scalar(self, sim_dict):
        print("(", str(self.pars['nx']), str(self.pars['alpha']), self.pars['beta'], self.pars['eta'], ") "
                                "Verror=", str(sim_dict["Verror"]),\
                                "Werror=", str(sim_dict["Werror"]),\
                                "time_compute=", str(sim_dict["time_compute"]))

    def compute_error(self, solver):
        vx, vy, vz, wx, wy, wz = util.read_data(cufcm_dir + 'data/simulation/simulation_data.dat', 0, self.pars['N'])
        if(solver == 0):
            vx_ref, vy_ref, vz_ref, wx_ref, wy_ref, wz_ref =\
                 util.read_data(save_directory + 'simulation_data' + util.parser(self.reference_pars) + '.dat', 0, self.reference_pars['N'])
        if(solver == 1):
            vx_ref, vy_ref, vz_ref, wx_ref, wy_ref, wz_ref =\
                 util.read_data(save_directory2 + 'simulation_data' + util.parser(self.reference_pars) + '.dat', 0, self.reference_pars['N'])
        return -2, -2


    def find_optimal(self, search_grid_width, tol, solver):
        w = search_grid_width
        loopshape = (w, w, w, w)
        time_compute_array = np.zeros(loopshape)
        Verror_array = np.zeros(loopshape)
        Werror_array = np.zeros(loopshape)
        alpha_array = np.zeros(loopshape)
        beta_array = np.zeros(loopshape)
        eta_array = np.zeros(loopshape)
        npts_array = np.zeros(loopshape)
        # Begin custom massive loop
        for i in range(w):
            for j in range(w):
                for k in range(w):
                    for npt in range(w):
                        self.pars['alpha']=      0.95 + 0.02*i
                        self.pars['beta']=       8 + 0.3*j
                        self.pars['eta']=        4.5 + 0.22*k
                        npts = util.compute_fastfcm_npts(self.pars['rh'])
                        self.pars['nx']=         npts
                        self.pars['ny']=         npts
                        self.pars['nz']=         npts

                        alpha_array[i, j, k] = self.pars['alpha']
                        beta_array[i, j, k] = self.pars['beta']
                        eta_array[i, j, k] = self.pars['eta']

                        if(sys.argv[1] == 'run'):
                            save_info_name, save_scalar_name, save_data_name = util.execute(self.pars, solver)

                        sim_dict = util.read_scalar(self.pars, solver)
                        if (sim_dict["Verror"] == -1):
                            sim_dict["Verror"], sim_dict["Werror"] = self.compute_error(solver)

                        self.print_scalar(sim_dict)
                        
        return 0, 0, 0, 0
