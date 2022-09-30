from cProfile import label
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import erf

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import subprocess
import os
import random

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
    ret = {}
    infoFile = open(fileName, 'r')
    lines = infoFile.readlines()
    for row in range(len(lines)):
        sep = find_pos(lines[row], symbol)
        ret[lines[row][:sep-1]] = float(lines[row][sep:])
    infoFile.close()
    return ret


def read_scalar(fileName, symbol='='):
    """Read scalar and create dict
    """
    ret = {}
    infoFile = open(fileName, 'r')
    lines = infoFile.readlines()
    for row in range(len(lines)):
        sep = find_pos(lines[row], symbol)
        ret[lines[row][:sep-1]] = float(lines[row][sep:])
    infoFile.close()
    return ret


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
        subprocess.call("cp " + "simulation_info " + save_info_name, shell=True)
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
    size_array = layer*50
    if(option[0] == 2):
        time_compute_layer = time_compute_array*layer
        min_value = time_compute_layer[time_compute_layer > 0].min()
        min_index = np.where(time_compute_layer == min_value)
        print("Min compute time for error=" + str(option[1]) + " is " + str(min_value) + " at " + str(min_index))
        print("Corresponding (alpha beta eta):" + str(alpha_array[min_index]) + ' '\
                                                + str(beta_array[min_index]) + ' '\
                                                + str(eta_array[min_index]) + ' ')

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
                        s=size_array, c=time_compute_array, cmap=cmap_name)
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

def monoExp(x, m, t):
    return m*np.exp(-t * x)

def monoGauss(x, m, t):
    return m*np.exp(-t * x**2)

def alpha_expr(tol):
    return np.sqrt(np.log10(tol/5.071050110367754)/(-10.224838558167573))

def beta_expr(tol, alpha):
    tol_list = np.array([2.7155083e-01, 7.4106760e-02, 5.9791320e-02, 1.2188930e-02,
                         8.8945900e-03, 1.3305500e-03, 8.9943000e-04, 9.7620000e-05, 
                         6.3130000e-05, 8.0200000e-06, 6.7500000e-06, 5.7200000e-06])
    ngd_list = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    for i, val in enumerate(tol_list):
        if tol >= val:
            return ngd_list[i]/alpha
    return ngd_list[-1]/alpha

def eta_expr(tol):
    return np.sqrt(np.log10(tol/6.300589575773176)/(-1.9832215241494984))

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
            return alpha_list[i], beta_list[i], 0
    return alpha_list[-1], beta_list[-1], 0