from pathlib import Path
import os


cufcm_dir = '/data/hs2216/CUFCM/'
info_file_name = "simulation_info"
date = "20230217"
fcm_directory = cufcm_dir + "data/simulation/" + date + "_fcm/"
fastfcm_directory = cufcm_dir + "data/simulation/" + date + "_fastfcm/"

global HIsolver
global save_directory
HIsolver = 1
if (HIsolver==0):
    save_directory = fcm_directory
if (HIsolver==1):
    save_directory = fastfcm_directory