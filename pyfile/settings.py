from pathlib import Path
import os


cufcm_dir = '/data/hs2216/CUFCM/'
info_file_name = "simulation_info_long"
fcm_directory = cufcm_dir + "data/simulation/20221118_fcm/"
fastfcm_directory = cufcm_dir + "data/simulation/20221118_fastfcm/"

solver = 1
if(solver == 0):
    save_directory = fcm_directory
if(solver == 1):
    save_directory = fastfcm_directory