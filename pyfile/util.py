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


def plot_3Dheatmap(alpha_array, beta_array, eta_array, error_array):
    # creating figures
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap_name = 'plasma'

    # setting color bar
    color_map = cm.ScalarMappable(cmap=cmap_name)
    color_map.set_array(error_array)
    
    # creating the heatmap
    img = ax.scatter(alpha_array, beta_array, eta_array, marker='s',
                    s=100, c=error_array, cmap=cmap_name)
    plt.colorbar(color_map)
    
    # adding title and labels
    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.savefig('img/plot.eps', format='eps')
    plt.show()

def plot_1D_fit(alpha_array, beta_array, eta_array, error_array, option="1n"):
    l = len(alpha_array[:, 0, 0])
    if(option[0] == "1"):
        data_array = alpha_array[:, -1, -1]
        y_array = error_array[:, -1, -1]
    if(option[0] == "2"):
        data_array = np.round(beta_array[-1, :, -1]*1.59)
        y_array = error_array[-1, :, -1]
        print(y_array)
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
        if tol > val:
            return ngd_list[i]/alpha
    return ngd_list[-1]/alpha

def eta_expr(tol):
    return np.sqrt(np.log10(tol/6.300589575773176)/(-1.9832215241494984))