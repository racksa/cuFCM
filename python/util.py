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
    save_info_name = directory + "simulation_info" + parser(idict)
    save_scalar_name = directory + "simulation_scalar" + parser(idict) + ".dat"
    save_data_name = directory + "simulation_data" + parser(idict) + ".dat"

    if(mode>0):
        subprocess.call("cp " + "simulation_info " + save_info_name, shell=True)
    if(mode>1):
        subprocess.call("cp " + "data/simulation/simulation_scalar.dat " + save_scalar_name, shell=True)
    if(mode>2):
        subprocess.call("cp " + "data/simulation/simulation_data.dat " + save_data_name, shell=True)