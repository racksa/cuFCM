from cProfile import label
from re import A
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erf

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import os
import random
import pandas as pd
from settings import *

def find_pos(line, symbol):
    """Find position of symbol
    """
    for char_pos, c in enumerate(line):
        if c == symbol:
            return char_pos+1
    print("Failed to find symbol in string")
    return 0


def find_key(key, lines):
    """Find the row the key belongs to
    """
    for i in range(len(lines)):
        if lines[i].find(key) == 0:
            return i
    print("Failed to find key in file")
    return -1


def replace(key, value, fileName):
    """Replace value in file by key
    """
    infoFile = open(fileName, 'r')
    lines = infoFile.readlines()
    row = find_key(key, lines)
    lines[row] = lines[row][:len(key)+1] + value + '\n'
    infoFile = open(fileName, 'w')
    infoFile.writelines(lines)
    infoFile.close()


def read_info(fileName, symbol='='):
    """Read info and create dict
    """
    ret_pardict = {}
    ret_filedict = {}
    infoFile = open(fileName, 'r')
    lines = infoFile.readlines()
    for row in range(len(lines)):
        if not lines[row][0] == '$' or lines[row][0] == ' ':
            sep = find_pos(lines[row], symbol)
            ret_pardict[lines[row][:sep-1]] = float(lines[row][sep:])
        elif lines[row][0] == '$':
            sep = find_pos(lines[row], symbol)
            ret_filedict[lines[row][:sep-1]] = lines[row][sep:].replace("\n", "")
    infoFile.close()
    return ret_pardict, ret_filedict


def read_scalar(idict, symbol='='):
    """Read scalar and create dict
    """
    fileName = save_directory + "simulation_scalar" + parser(idict) + ".dat"
    if not os.path.isfile(fileName):
        print('Error file name' + fileName)
        return False
    ret = {}
    infoFile = open(fileName, 'r')
    lines = infoFile.readlines()
    for row in range(len(lines)):
        sep = find_pos(lines[row], symbol)
        ret[lines[row][:sep-1]] = float(lines[row][sep:])
    infoFile.close()
    return ret


def write_scalar(idict, sim_dict):
    """Write scalar and create dict
    """
    fileName = save_directory + "simulation_scalar" + parser(idict) + ".dat"
    for key in sim_dict:
        replace(key, str(sim_dict[key]), fileName)


def file_exists(idict):
    """Check if file exists
    """
    fileName = save_directory + "simulation_scalar" + parser(idict) + ".dat"
    if os.path.exists(fileName):
        return True
    return False


def read_velocity(filePath, t, N):
    '''Read simulation data at time frame t
    '''
    read_frame = pd.read_csv(filePath, delimiter=' ', header=None, skiprows=int(t*(N+1)), nrows=N)

    vx = read_frame[9]
    vy = read_frame[10]
    vz = read_frame[11]
    wx = read_frame[12]
    wy = read_frame[13]
    wz = read_frame[14]
    return vx, vy, vz, wx, wy, wz

def separate_datafile(filePath):
    '''Separate quantities into different files
    '''
    infoFile = open(filePath, 'r')
    lines = infoFile.readlines()
    N = len(lines)-2
    infoFile.close()

    read_frame = pd.read_csv(filePath, delimiter=' ', header=None, skiprows=int(0*(N+1)), nrows=N)

    x = read_frame[0]
    y = read_frame[1]
    z = read_frame[2]
    fx = read_frame[3]
    fy = read_frame[4]
    fz = read_frame[5]
    tx = read_frame[6]
    ty = read_frame[7]
    tz = read_frame[8]

    infoFile = open(filePath, 'r')
    pos_lines = list()
    force_lines = list()
    torque_lines = list()
    for row in range(N):
        pos_string = str(x[row]) + ' ' + str(y[row]) + ' ' + str(z[row]) + '\n'
        pos_lines.append(pos_string)
        force_string = str(fx[row]) + ' ' + str(fy[row]) + ' ' + str(fz[row]) + '\n'
        force_lines.append(force_string)
        torque_string = str(tx[row]) + ' ' + str(ty[row]) + ' ' + str(tz[row]) + '\n'
        torque_lines.append(torque_string)

    infoFile = open(cufcm_dir + "data/init_data/artificial/output_pos_data.dat", 'w')
    infoFile.writelines(pos_lines)
    infoFile.close()
    infoFile = open(cufcm_dir + "data/init_data/artificial/output_force_data.dat", 'w')
    infoFile.writelines(force_lines)
    infoFile.close()
    infoFile = open(cufcm_dir + "data/init_data/artificial/output_torque_data.dat", 'w')
    infoFile.writelines(torque_lines)
    infoFile.close()


def parser(idict):
    if(not type(idict) is dict):
        print("Not a dict object")
        return ''
    def smart_str(value):
        if(int(value) == float(value)):
            return str(int(value))
        else:
            return "{:.4f}".format(value)
    ret = ''
    for key in idict:
        ret += '_' + key + smart_str(idict[key])
    return ret


def savefile(idict, directory, mode=1):
    save_info_name = directory + "simulation_info" + parser(idict) + ".dat"
    save_scalar_name = directory + "simulation_scalar" + parser(idict) + ".dat"
    save_data_name = directory + "simulation_data" + parser(idict) + ".dat"

    if(mode>0):
        subprocess.call("cp " + info_file_name + " " + save_info_name, shell=True)
    if(mode>1):
        subprocess.call("cp " + "data/simulation/simulation_scalar.dat " + save_scalar_name, shell=True)
    if(mode>2):
        subprocess.call("cp " + "data/simulation/simulation_data.dat " + save_data_name, shell=True)

    return save_info_name, save_scalar_name, save_data_name


def plot_3Dheatmap(alpha_array, beta_array, eta_array, error_array, time_compute_array, option=[1, 1]):
    # creating figures
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap_name = 'plasma'

    # setting error layer
    layer = layer_array(error_array, option[1])
    markersize_array = layer*50
    if(option[0] == 2):
        time_compute_layer = time_compute_array*layer
        min_value = time_compute_layer[time_compute_layer > 0].min()
        min_index = np.where(time_compute_layer == min_value)
        print("Min compute time for error=" + str(option[1]) + " is " + str(min_value) + " at " + str(min_index))
        print("Corresponding (alpha beta eta):" + str(alpha_array[min_index][0]) + ' '\
                                                + str(beta_array[min_index][0]) + ' '\
                                                + str(eta_array[min_index][0]) + ' ')

    # setting color bar
    color_map = cm.ScalarMappable(cmap=cmap_name)
    if(option[0] == 1):
        save_name = 'error_heatmap.eps'
        color_map.set_array(np.log10(error_array))
    if(option[0] == 2):
        save_name = 'time_layer' + str(option[1]) + '.eps'
        color_map.set_array(time_compute_array)
    cbar=plt.colorbar(color_map)
    
    # creating the heatmap
    if(option[0] == 1):
        img = ax.scatter(alpha_array, beta_array, eta_array, marker='s',
                        s=50, c=np.log10(error_array), cmap=cmap_name)
        cbar.set_label("log10(Error)")
        ax.set_title("Error heatmap")
    if(option[0] == 2):
        img = ax.scatter(alpha_array, beta_array, eta_array, marker='s',
                        s=markersize_array, c=time_compute_array, cmap=cmap_name)
        cbar.set_label("Compute time")
        ax.set_title("Compute time for Error=" + str(option[1]))

    # adding title and labels
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$\eta$')
    plt.savefig('img/'+save_name, format='eps')
    plt.show()


def plot_1D_fit(alpha_array, beta_array, eta_array, error_array, option="1n"):
    l = len(alpha_array[:, 0, 0])
    if(option[0] == "1"):
        data_array = alpha_array[:, -1, -1]
        y_array = error_array[:, -1, -1]
    if(option[0] == "2"):
        data_array = np.round(beta_array[-1, :, -1]*1.59)
        y_array = error_array[-1, :, -1]
    if(option[0] == "3"):
        data_array = eta_array[-1, -1, :]
        y_array = error_array[-1, -1, :]
    if(option[0] == "4"):
        data_array = [alpha_array[i, i, i] for i in range(l)]
        y_array = [error_array[i, i, i] for i in range(l)]
        
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    # plot data
    ax.plot(data_array, y_array, marker='o', label="data")

    if(option[1] == "f"):
        x_fit = np.linspace(data_array[0], data_array[-1], 20)
        if(option[0] == "1"):
            params, cv = curve_fit(monoGauss, data_array, y_array, (1, 10))
            m, t = params
            ax.plot(x_fit, monoGauss(x_fit, m, t), label="fit curve")
        if(option[0] == "3"):
            params, cv = curve_fit(monoExp, data_array, y_array, (1, 10))
            m, t = params
            ax.plot(x_fit, monoExp(x_fit, m, t), label="fit curve")
        if(option[0] == "4"):
            params, cv = curve_fit(monoExp, data_array, y_array, (1, 10))
            m, t = params
            ax.plot(x_fit, monoExp(x_fit, m, t), label="fit curve")
        print(params)

    ax.legend()

    # adding title and labels
    ax.set_title("Fit")
    ax.set_xlabel('data')
    ax.set_ylabel('error')
    ax.set_yscale('log')
    plt.savefig('img/plot'+option+'.eps', format='eps')
    plt.show()


def plot_npts(npts_array, error_array, time_compute_array):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax2 = ax.twinx()
    
    # plot data
    min_time = time_compute_array.min()
    min_cor_npts = npts_array[np.where(time_compute_array == min_time) ]
    min_cor_error = error_array[np.where(time_compute_array == min_time) ]
   
    print("Best timing: " + str(min_time))
    print("Corresponding NPTS: " + str(min_cor_npts))
    print("Corresponding error: " + str(min_cor_error))
    print("Corresponding PTPS: " + str(500000/min_time))
    ax.plot(npts_array, error_array, marker='o', c='r', label="% Error")
    ax2.plot(npts_array, time_compute_array, marker='+', label="Compute time")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # adding title and labels
    ax.set_title("% Error vs. npts")
    ax.set_xlabel('Npts')
    ax.set_ylabel('% Error')
    ax2.set_ylabel('Compute time s')
    plt.savefig('img/npts.eps', format='eps')
    plt.show()


def plot_n(n_array, time_compute_array, time_compute_array2):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ptps_array = n_array/time_compute_array
    ptps_array2 = n_array/time_compute_array2
    
    fcm_cutoff_point = len(n_array)
    for i, t in enumerate(time_compute_array):
        if (i > 0 and abs(t - time_compute_array[i-1]) < 1.e-5):
            fcm_cutoff_point = i
            break

    ax.plot(n_array[:fcm_cutoff_point], ptps_array[:fcm_cutoff_point], marker='o', c='r', label="FCM")
    ax.plot(n_array, ptps_array2, marker='+', c='b', label="Fast FCM")
    ax.legend(loc='upper right')
    
    # adding title and labels
    ax.set_title("PTPS vs. N")
    ax.set_xlabel('N')
    ax.set_ylabel('PTPS')
    # ax2.set_ylabel('Compute time s')
    plt.savefig('img/PTPS_N.eps', format='eps')
    plt.show()


def layer_array(error_array, tol):
    ret = np.ones(np.shape(error_array))
    for i in range(np.shape(error_array)[0]):
        for j in range(np.shape(error_array)[1]):
            for k in range(np.shape(error_array)[2]):
                if error_array[i][j][k] > tol:
                    ret[i][j][k] = 0
                elif (i*j*k>0) and (error_array[i-1][j][k]<= tol and \
                                    error_array[i][j-1][k]<= tol and \
                                    error_array[i][j][k-1]<= tol ):
                    ret[i][j][k] = 0
                else:
                    ret[i][j][k] = 1
    return ret


def filter_array(error_array, tol):
    ret = np.ones(np.shape(error_array))
    for i in range(np.shape(error_array)[0]):
        for j in range(np.shape(error_array)[1]):
            for k in range(np.shape(error_array)[2]):
                for l in range(np.shape(error_array)[3]):
                    if error_array[i][j][k][l] > tol:
                        ret[i][j][k][l] = 0.
                    else:
                        ret[i][j][k][l] = 1.
    return ret


def filter_array_1D(error_array, tol):
    ret = np.ones(len(error_array))
    for i in range(len(error_array)):
        if error_array[i] > tol:
            ret[i] = 0.
        else:
            ret[i] = 1.
    return ret


def compute_rad(N, volume_frac):
    return (6*np.pi**2*volume_frac/N)**(1./3.)

def compute_phi(N, rad):
    return (4./3.*np.pi)*N*rad**3/(2.*np.pi)**3

def compute_N(phi, rad):
    return int(phi*(2*np.pi/rad)**3/(4./3.*np.pi))


def compute_fastfcm_npts(rad):
    return 2*int(0.02609300415934458 * 256. / rad /2)

def monoExp(x, m, t):
    return m*np.exp(-t * x)

def monoGauss(x, m, t):
    return m*np.exp(-t * x**2)

# def alpha_expr(tol):
#     return np.sqrt(np.log10(tol/5.071050110367754)/(-10.224838558167573))

# def beta_expr(tol, alpha):
#     tol_list = np.array([2.7155083e-01, 7.4106760e-02, 5.9791320e-02, 1.2188930e-02,
#                          8.8945900e-03, 1.3305500e-03, 8.9943000e-04, 9.7620000e-05, 
#                          6.3130000e-05, 8.0200000e-06, 6.7500000e-06, 5.7200000e-06])
#     ngd_list = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
#     for i, val in enumerate(tol_list):
#         if tol >= val:
#             return ngd_list[i]/alpha
#     return ngd_list[-1]/alpha

# def eta_expr(tol):
#     return np.sqrt(np.log10(tol/6.300589575773176)/(-1.9832215241494984))

def par_given_error(tol):
    tol_list = np.array([5.e-3, 1.e-3, 5.e-4, 1.e-4, 5.e-5, 1.e-5])
    alpha_list = np.array([0.95, 0.97, 0.97, 1.09, 1.09, 1.17])
    beta_list = np.array([9.8, 8.9, 10.7, 11.3, 10.1, 10.7 ])
    eta_list = np.array([4.5, 4.94, 5.6, 6.04, 6.7, 7.14])
    for i, val in enumerate(tol_list):
        if tol >= val:
            return alpha_list[i], beta_list[i], eta_list[i]
    return alpha_list[-1], beta_list[-1], eta_list[-1]

def fcm_par_given_error(tol, rh):
    tol_list = np.array([5.e-3, 1.e-3, 1.e-4, 1.e-5])
    alpha_list = np.array([0.98, 1.1, 1.22, 1.31])
    beta_list = np.array([7.2, 8.0, 8.8, 9.6 ])
    for i, val in enumerate(tol_list):
        if tol >= val:
            return alpha_list[i], beta_list[i], 6
    return alpha_list[-1], beta_list[-1], 6

def par_reference(rad):
    return 1.3, 11., 8., compute_fastfcm_npts(rad)

def execute(dicts, solver=1, mode=2):
        for dic in dicts:
            for key in dic:
                replace(key, str(dic[key]), info_file_name)
        pars = dicts[0] #first dictionary should always be the parameter dictionary!!
        if(solver == 0):
            subprocess.call(cufcm_dir + "bin/FCM", shell=True)
            return savefile(pars, save_directory, mode)
        if(solver == 1):
            subprocess.call(cufcm_dir + "bin/CUFCM", shell=True)
            return savefile(pars, save_directory, mode)
        

def execute_random_generator(pars):
    for key in pars:
        replace(key, str(pars[key]), info_file_name)
    subprocess.call(cufcm_dir + "bin/RANDOM", shell=True)

def percentage_error_magnitude(x, xref):
    return np.mean( modulus(x-xref) / modulus(xref) )

def modulus(vec):
    return np.sqrt(np.sum(vec*vec, 0))

color_codex = {0: 'r',
               1: 'b',
               2: 'g',
               3: 'y',
               4: 'c',
               5: 'grey',
               6: 'black',
               7: 'brown',
               8: 'orange',
               9: 'purple',
               10: 'silver',
               11: 'aqua',
               12: 'gold',
               13: 'fuchsia'}