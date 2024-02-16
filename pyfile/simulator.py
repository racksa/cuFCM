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
np.set_printoptions(threshold=sys.maxsize)

class SIM:

    def __init__(self):
        pardict, filedict = util.read_info(info_file_name)
        # Initialise parameters
        pardict['repeat']=     50
        pardict['prompt']=     -1
        pardict['dt']=         0.1
        pardict['Fref']=       pardict['rh']
        pardict['packrep']=    1000
        pardict['boxsize']=    500

        self.pars = pardict.copy()
        self.datafiles = filedict.copy()
        self.reference_pars = pardict.copy()

        # self.search_grid_shape = (17, 10, 1, 1) # alpha, beta, eta, npts
        self.search_grid_shape = (1, 1, 1, 50) # alpha, beta, eta, npts

        self.na = 12
        self.nphi = 21
        
        loopshape = (self.na, self.nphi)
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

        for i in range(self.na):
            for j in range(self.nphi):
                phi=                        0.01 + 0.01*j
                # self.pars['rh']=            2.0 + (4./12)*i
                self.pars['rh']=            1.9894367886486917 + 0.477464829275686*i
                self.pars['N']=             util.compute_N(phi, self.pars['rh'], self.pars['boxsize'])

                ###### temporary ######
                # self.pars['N']=2
                ###### temporary ######

                phi=                        util.compute_phi(self.pars['N'], self.pars['rh'], self.pars['boxsize'])
                self.pars['Fref']=          self.pars['rh']

                self.phi_array[i, j] = phi
                self.rh_array[i, j] = self.pars['rh']
                self.n_array[i, j] = self.pars['N']

                self.print_siminfo(i, j)
                if(sys.argv[1] == 'run'):
                    util.execute_random_generator(self.pars)
                    self.get_reference()

                self.pars['alpha'], self.pars['beta'], self.pars['eta'], npts, optimal_Verror, optimal_Werror, optimal_time = self.find_optimal(2.e-6)
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
        print("Generating reference data")
        self.reference_pars=                self.pars.copy()
        self.reference_pars['alpha']        = 2.0
        self.reference_pars['beta']         = 32.0
        self.reference_pars['eta']          = round(24.0, 1)
        # npts                                = min( int(0.026/self.reference_pars['rh'] * 270 /2) * 2, 460)
        npts = 400
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
        sigma_ratio_array = np.zeros(self.search_grid_shape)
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
                        self.pars['alpha']=      1.2 + 0.02*i
                        self.pars['beta']=       (12. + 1*j )
                        self.pars['eta']=        5.8 #+ np.exp(-8e-6*self.pars['N'])

                        if(HIsolver==1):
                            npts = min(60 + 10*l, int(self.pars['alpha']*self.pars['boxsize']/(self.pars['rh']/np.sqrt(np.pi)) /2)*2 )
                        if(HIsolver==0):
                            npts = 60 + 10*l
                        self.pars['nx']=         npts
                        self.pars['ny']=         npts
                        self.pars['nz']=         npts

                        if HIsolver == 0:
                            dx = self.pars['boxsize']/npts
                            self.pars['alpha']= (self.pars['rh']/np.sqrt(np.pi))/dx
                            self.pars['eta']= self.pars['eta']/self.pars['alpha']

                        # temporary #############################
                        # Sigma = (self.pars['alpha']*self.pars['boxsize']/self.pars['nx'])
                        # self.pars['eta']=        (2+2*k)/Sigma
                        # self.pars['eta']=       0.1/Sigma
                        # temporary #############################

                        if(sys.argv[1] == 'run'):
                            util.execute([self.pars, self.datafiles], solver=HIsolver, mode=2)

                        sim_dict = util.read_scalar(self.pars)
                        if(sim_dict):
                            if (sim_dict["Verror"] == -1 and sys.argv[1] == 'run'):
                                sim_dict["Verror"], sim_dict["Werror"] = self.compute_error()

                            time_compute_array[i, j, k, l] = sim_dict["time_compute"]
                            sigma_ratio_array[i, j, k, l] = (self.pars['alpha']*self.pars['boxsize']/self.pars['nx'])/(self.pars['rh']/np.sqrt(np.pi))
                            Verror_array[i, j, k, l] = sim_dict["Verror"]
                            Werror_array[i, j, k, l] = sim_dict["Werror"]
                            alpha_array[i, j, k, l] = self.pars['alpha']
                            beta_array[i, j, k, l] = self.pars['beta']
                            eta_array[i, j, k, l] = self.pars['eta']
                            npts_array[i, j, k, l] = self.pars['nx']

                            self.print_scalar(sim_dict)
                            util.write_scalar(self.pars, sim_dict)

        ptps_array = self.pars['N']/time_compute_array
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

        # print('N=', self.pars['N'])
        # print('sigma_ratio_array=np.array(' + np.array2string(sigma_ratio_array[0,0,:,:], separator=", ") + ')')
        # print('eta_array=np.array(' + np.array2string(eta_array[0,0,:,:], separator=", ") + ')')
        # print('error_array=np.array(' + np.array2string(Verror_array[0,0,:,:], separator=", ") + ')')

        print('sigma_ratio_array=np.array(' + np.array2string(sigma_ratio_array[0,0,0,:], separator=", ") + ')')
        print('ptps_array=np.array(' + np.array2string(ptps_array[0,0,0,:], separator=", ") + ')')
        
        # print('alpha_array=np.array(' + np.array2string(alpha_array[:,:,0,0], separator=", ") + ')')
        # print('beta_array=np.array(' + np.array2string(beta_array[:,:,0,0], separator=", ") + ')')
        # print('error_array=np.array(' + np.array2string(Verror_array[:,:,0,0], separator=", ") + ')')

        # print('eta_array=' + np.array2string(sigma_ratio_array[0,0,0,:], separator=", "))
        # print('error_array=' + np.array2string(Verror_array[0,0,0,:], separator=", "))

        optimal_Verror = Verror_array[min_index][0]
        optimal_Werror = Werror_array[min_index][0]
        optimal_alpha = alpha_array[min_index][0]
        optimal_beta = beta_array[min_index][0]
        optimal_eta = eta_array[min_index][0]
        optimal_npts = npts_array[min_index][0]


        print("Min compute time for error=" + str(tol) + " is " + str(min_time))
        print("Corresponding (alpha beta eta npts):" + '{:.2f}'.format(optimal_alpha) + ' '\
                                                + '{:.2f}'.format(optimal_beta) + ' '\
                                                + '{:.2f}'.format(optimal_eta) + ' '\
                                                + '{:.2f}'.format(optimal_npts) + ' ')
                        
        return optimal_alpha, optimal_beta, optimal_eta, optimal_npts, optimal_Verror, optimal_Werror, min_time

    def run_test(self):
        self.datafiles['$posfile'] = './test/test_data/pos-N500000-rh02609300-2.dat'
        self.datafiles['$forcefile'] = './test/test_data/force-N500000-rh02609300.dat'
        self.datafiles['$torquefile'] = './test/test_data/force-N500000-rh02609300-2.dat'
        self.pars['checkerror'] = 1

        self.pars['N']=          500000
        self.pars['rh']=         0.02609300415934458
        self.pars['alpha']=      1.0
        self.pars['beta']=       9.0
        self.pars['eta']=        4.8
        npts = 480  # Regular FCM

        self.pars['nx']=         npts
        self.pars['ny']=         npts
        self.pars['nz']=         npts
        self.pars['Fref']=       1.0
        self.pars['repeat']=     50
        self.pars['prompt']=     10
        self.pars['boxsize']=    np.pi*2

        if HIsolver == 0:
            dx = self.pars['boxsize']/npts
            self.pars['alpha']= (self.pars['rh']/np.sqrt(np.pi))/dx
            self.pars['eta']= self.pars['eta']/self.pars['alpha']

        # util.execute([self.pars, self.datafiles], solver=0, mode=3)

        npts = 320  # Fast FCM
        self.pars['nx']=         npts
        self.pars['ny']=         npts
        self.pars['nz']=         npts
        util.execute([self.pars, self.datafiles], solver=1, mode=3)

        # self.plot_pie_chart_of_time()
    
    def run_single(self):
        self.datafiles['$posfile'] = './data/init_data/artificial/artificial_pos.dat'
        self.datafiles['$forcefile'] = './data/init_data/artificial/artificial_force.dat'
        self.datafiles['$torquefile'] = './data/init_data/artificial/artificial_torque.dat'

        self.datafiles['$posfile'] = './data/init_data/new/pos_data.dat'
        self.datafiles['$forcefile'] = './data/init_data/new/force_data.dat'
        self.datafiles['$torquefile'] = './data/init_data/new/torque_data.dat'
        self.pars['checkerror'] = 0

        self.pars['N']=          2
        self.pars['rh']=         1.0
        self.pars['alpha']=      1.0
        self.pars['beta']=       10.0
        self.pars['eta']=        8.0
        self.pars['nx']=         60
        self.pars['ny']=         60
        self.pars['nz']=         60
        self.pars['Fref']=       1.0
        self.pars['repeat']=     1
        self.pars['prompt']=     10
        self.pars['boxsize']=    400

        util.execute([self.pars, self.datafiles], solver=HIsolver, mode=3)
        
    def run_two_particle(self):
        self.datafiles['$posfile'] = './data/init_data/new/pos_data.dat'
        self.datafiles['$forcefile'] = './data/init_data/new/force_data.dat'
        self.datafiles['$torquefile'] = './data/init_data/new/torque_data.dat'
        self.pars['checkerror'] = 0

        r_list = 1e-1 + np.linspace(1, 10, 11)
        npts_list = np.arange(60, 480, 20)
        for r in r_list:
            for npts in npts_list:
                self.pars['alpha']=      2.0
                self.pars['beta']=       (15.)

                # npts = min(60 + 20*l, int(self.pars['boxsize']/(self.pars['rh']/np.sqrt(np.pi)) /2)*2 )
                self.pars['nx']=         npts
                self.pars['ny']=         npts
                self.pars['nz']=         npts

                if HIsolver == 0:
                    dx = self.pars['boxsize']/npts
                    self.pars['alpha']= (self.pars['rh']/np.sqrt(np.pi))/dx
                    self.pars['eta']= self.pars['eta']/self.pars['alpha']

                # temporary #############################
                Sigma = (self.pars['alpha']*self.pars['boxsize']/self.pars['nx'])
                # self.pars['eta']=        (1+1*k)/Sigma
                self.pars['eta']=       0.1/Sigma
                # temporary #############################

                util.execute([self.pars, self.datafiles], solver=HIsolver, mode=3)

                sim_dict = util.read_scalar(self.pars)
                if(sim_dict):
                    if (sim_dict["Verror"] == -1 and sys.argv[1] == 'run'):
                        sim_dict["Verror"], sim_dict["Werror"] = self.compute_error()

    def evaluate_flow_field(self):
        dir = './data/flow_data/'
        for file in os.listdir(dir):
            if file.endswith('flow_pos.dat'):
                self.datafiles['$posfile'] = dir + file
            if file.endswith('flow_force.dat'):
                self.datafiles['$forcefile'] = dir + file
            if file.endswith('flow_torque.dat'):
                self.datafiles['$torquefile'] = dir + file
        self.pars['checkerror'] = 0
        self.pars['repeat']=     1
        self.pars['prompt']=     10

        self.pars['rh']=         1

        # self.pars['N']=          22*1024
        # self.pars['beta']=       10
        # self.pars['nx']=         1280
        # self.pars['ny']=         1280
        # self.pars['nz']=         40     
        # self.pars['boxsize']=    1280

        self.pars['N']=          20*64
        self.pars['beta']=       20
        self.pars['nx']=         480
        self.pars['ny']=         480
        self.pars['nz']=         60
        self.pars['boxsize']=    640

        Lx = self.pars['boxsize']
        Ly = Lx/self.pars['nx']*self.pars['ny']
        Lz = Lx/self.pars['nx']*self.pars['nz']

        util.execute([self.pars, self.datafiles], solver=2, mode=3)

        flow_x_f = open('./data/simulation/flow_x.dat', "r")
        flow_y_f = open('./data/simulation/flow_y.dat', "r")
        flow_z_f = open('./data/simulation/flow_z.dat', "r")
        pos_f = open(self.datafiles['$posfile'], "r")
        pos = np.zeros((self.pars['N'], 3))
        for i in range(self.pars['N']):
            pos[i] = np.array(pos_f.readline().split(), dtype=float)
        flow_x = np.array(flow_x_f.readline().split(), dtype=float)
        flow_y = np.array(flow_y_f.readline().split(), dtype=float)
        flow_z = np.array(flow_z_f.readline().split(), dtype=float)
        
        def reshape_func(flow):
            return np.reshape(flow, (self.pars['nz'], self.pars['ny'], self.pars['nx'])) # z-major
        
        sigma = float(self.pars['rh'])/np.pi**.5
        def B(r):
            return 1./(8*np.pi*r**3)*( (1-3*sigma**2/r**2)*erf(r/(sigma*2**.5)) + (6*sigma/r)*(2*np.pi)**(-0.5)*np.exp(-r**2/(2*sigma**2)) )

        def A(r):
            return 1./(8*np.pi*r)*( (1+sigma**2/r**2)*erf(r/(sigma*2**.5)) - (2*sigma/r)*(2*np.pi)**(-0.5)*np.exp(-r**2/(2*sigma**2)) )

        def u(x):
            x = np.array([x])
            r = np.linalg.norm(x)
            # return np.matmul( np.identity(3)/r + (x.transpose()@x)/r**3, np.array([1, 0, 0]))/(8*np.pi)
            return np.matmul( A(r)*np.identity(3) + B(r)*(x.transpose()@x), np.array([1, 0, 0]))

        flow_x = reshape_func(flow_x)
        flow_y = reshape_func(flow_y)
        flow_z = reshape_func(flow_z)

        X = np.linspace(0, Lx, self.pars['nx'])
        Y = np.linspace(0, Ly, self.pars['ny'])
        Z = np.linspace(0, Lz, self.pars['nz'])

        W = 0.0
        dx = Lx/self.pars['nx']
        nxh = int(self.pars['nx']/2)
        nyh = int(self.pars['ny']/2)
        nzh = int(self.pars['nz']/2)

        z = nzh + int(np.floor(20/dx))
        
        # flow_expression_x = np.zeros((self.pars['ny'], self.pars['nx']))
        # flow_expression_y = np.zeros((self.pars['ny'], self.pars['nx']))
        # for i in range(self.pars['nx']):
        #     for j in range(self.pars['ny']):
        #         rx = dx*i
        #         ry = dx*j
        #         rz = pos[0][2]
        #         flow_expression_x[j][i], flow_expression_y[j][i], vz = u(np.array([rx, ry, rz])-pos[0])
        # print(flow_expression_x[nyh])
        # print(flow_x[z][nyh])
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     diff = np.abs((flow_expression_x - flow_x[nzh]))
        # diff[np.isnan(diff)] = 0
        # diff[np.isinf(diff)] = 0
        # print('mean diff=', np.mean(diff))


        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(pos[:,0], pos[:,1], c='r')

        # circle = plt.Circle((pos[:,0], pos[:,1]), self.pars['rh'], color='r')
        # ax.add_patch(circle)
        # ax.set_ylim((0,100))

        ax.streamplot(X, Y, flow_x[z]-W, flow_y[z])
        # ax.streamplot(X, Y, flow_expression_x-W, flow_expression_y)

        qfac = 10
        ax.quiver(X[::qfac], Y[::qfac], flow_x[z, ::qfac,::qfac]-W, flow_y[z,::qfac,::qfac])
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig('img/flow_field.eps', format='eps')
        plt.show()



    def print_scalar(self, sim_dict):
        if(self.siminfo):
            print("(", '{:.0f}'.format(self.pars['nx']), '{:.2f}'.format(self.pars['alpha']), '{:.0f}'.format(self.pars['beta']), '{:.2f}'.format(self.pars['eta']), ") "
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
        vx, vy, vz, wx, wy, wz = util.read_velocity(cufcm_dir + 'data/simulation/simulation_data.dat', 0, int(self.pars['N']))
        # vx_ref, vy_ref, vz_ref, wx_ref, wy_ref, wz_ref =\
        #         util.read_velocity(fastfcm_directory + 'simulation_data' + util.parser(self.reference_pars) + '.dat', 0, int(self.reference_pars['N']))
        vx_ref, vy_ref, vz_ref, wx_ref, wy_ref, wz_ref =\
                util.read_velocity(save_directory + 'simulation_data' + util.parser(self.reference_pars) + '.dat', 0, int(self.reference_pars['N']))

        v_array = np.array([vx, vy, vz])
        vref_array = np.array([vx_ref, vy_ref, vz_ref])
        w_array = np.array([wx, wy, wz])
        wref_array = np.array([wx_ref, wy_ref, wz_ref])
        
        return util.percentage_error_magnitude(v_array, vref_array), util.percentage_error_magnitude(w_array, wref_array)


    def analyse(self):
        self.load_optimal_arrays()

        self.na, self.nphi = np.shape(self.optimal_alpha_array)

        print(np.shape(self.optimal_alpha_array))
        print('Optimal parameters:')
        optimal_parameters = [[i, j, self.optimal_alpha_array[i, j], self.optimal_beta_array[i, j],\
                self.optimal_eta_array[i, j], self.optimal_npts_array[i, j]] \
                for i in range(self.na) for j in range(self.nphi)]
        optimal_times = [self.optimal_time_compute_array[i, j] \
                for i in range(self.na) for j in range(self.nphi)]
        for i, pars in enumerate(optimal_parameters):
            print(pars, 'PTPS=', self.pars['N']/optimal_times[i])
            if ((i+1)%self.nphi == 0):
                print('--------------------')

        print(self.optimal_time_compute_array)

    def analyse_and_plot_both(self):
        fig = plt.figure(figsize=(4.8, 3.6))
        ax = fig.add_subplot(1,1,1)
        fig2 = plt.figure(figsize=(4.8, 3.6))
        ax2 = fig2.add_subplot(1,1,1)

        import matplotlib
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

        fcm_ptps_array = np.zeros(np.shape(self.phi_array[0]))
        ffcm_ptps_array = np.zeros(np.shape(self.phi_array[0]))

        fcm_ptps_array_list = list()
        
        global fcm_directory
        global fastfcm_directory

        # Old one 10^-4
        fcm_directory = cufcm_dir + "data/simulation/20221129_fcm/"
        fastfcm_directory = cufcm_dir + "data/simulation/20221129_fastfcm/"
        self.pars['boxsize'] = 6.283185307179586
        plot_interval = 4

        # # New one 10^-6
        fcm_directory = cufcm_dir + "data/simulation/20240212_fcm/"
        fastfcm_directory = cufcm_dir + "data/simulation/20240212_fastfcm/"
        self.pars['boxsize'] = 500.0
        plot_interval = 4
        

        for j in range(2):
            self.mod_solver(j)
            self.analyse()

            # print('^^^^^^^^^', self.rh_array)

            linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot', '' ]
            marker_list = ['', '', '', '', '+']
            for i in range(0, len(self.rh_array), plot_interval):
                print("========", i, len(self.rh_array))
                # print(self.rh_array[i][0]/self.pars['boxsize'])
                if (HIsolver==0):
                    label = 'FCM a/L=' + str(round(self.rh_array[i][0]/self.pars['boxsize'], 3))
                    marker = '+'
                    linestyle='solid'
                    fcm_ptps_array = self.n_array[i]/self.optimal_time_compute_array[i]
                    fcm_ptps_array_list.append(fcm_ptps_array)
                if (HIsolver==1):
                    label = 'F-FCM a/L=' + str(round(self.rh_array[i][0]/self.pars['boxsize'], 3))
                    marker = ','
                    linestyle='dashed'
                    ffcm_ptps_array = self.n_array[i]/self.optimal_time_compute_array[i]
                
                ptps_array = self.n_array[i]/self.optimal_time_compute_array[i]
                ax.plot(self.phi_array[i], ptps_array, marker=marker, linestyle=linestyle, c=util.color_codex[i], label=label)
                if(j==1):
                    ratio = ffcm_ptps_array/fcm_ptps_array_list[int(i/plot_interval)]
                    ax2.axhline(y = 1, color = 'grey', linestyle = '-.', lw=0.5)
                    ax2.plot(self.phi_array[i], ratio, linestyle=linestyle_list[int(i/plot_interval)], c='black', label=r'$a/L=$' + str(round(self.rh_array[i][0]/self.pars['boxsize'], 3)))
                    # plot cross points
                    idx = np.argwhere(np.diff(np.sign(ratio - 1))).flatten()[0]
                    lenx = (self.phi_array[i][idx+1] - self.phi_array[i][idx])
                    leny = (ratio[idx+1] - ratio[idx])
                    grad = leny/lenx
                    interception_x = self.phi_array[i][idx] + (1-ratio[idx])/grad
                    # print('intercept', interception_x)

                    # ax2.scatter(interception_x, 1, marker = 'x', c='black', zorder=10)
                    # ax.scatter(aL_array, crossover_array, marker = '+', s=100, color='red', label='Data', zorder=10)

        # adding title and labels
        # ax.set_title(r"PTPS vs. $\phi$")
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel('PTPS')
        ax.legend()
        # ax.set_xscale('log')
        ax2.set_xlabel(r'$\phi$')
        ax2.set_ylabel(r'$PTPS_{ffcm}/PTPS_{fcm}$')
        ax2.set_yscale('log')
        ax2.legend()
        fig.savefig('img/ptps_combined.pdf', bbox_inches = 'tight', format='pdf')
        fig.savefig('img/ptps_combined.png', bbox_inches = 'tight', format='png')
        fig2.savefig('img/ptps_combined_ratio.pdf', bbox_inches = 'tight', format='pdf')
        fig2.savefig('img/ptps_combined_ratio.png', bbox_inches = 'tight', format='png')
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