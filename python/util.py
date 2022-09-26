from mpl_toolkits.mplot3d import Axes3D
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


def plot_3Dheatmap(alpha_array, beta_array, eta_array, error_array ):
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
    
    # displaying plot
    plt.show()

def plot_1D_fit(data_array, error_array):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    # creating the heatmap
    img = ax.line(data_array, error_array, marker='s', s=100)
    
    # adding title and labels
    ax.set_title("Fit")
    ax.set_xlabel('data')
    ax.set_ylabel('error')

    # displaying plot
    plt.show()