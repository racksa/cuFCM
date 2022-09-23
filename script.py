import sys
import subprocess
import numpy as np
from python import util

def run():
    info_file_name = "simulation_info"
    par_dict = util.read_info(info_file_name)
    save_directory = "data/simulation/test/"

    # Begin custom massive loop
    for i in range(1):
        for j in range(1):
            for k in range(1):
                
                if(sys.argv[1] == 'run'):

                
                    npts = 400
                    par_dict['N']=          500000.0
                    par_dict['rh']=         0.02609300415934458
                    par_dict['alpha']=      1.0/(2*np.pi/npts) * par_dict['rh']/np.sqrt(np.pi)
                    par_dict['beta']=       15
                    par_dict['eta']=        0.2
                    par_dict['nx']=         npts
                    par_dict['ny']=         npts
                    par_dict['nz']=         npts
                    par_dict['repeat']=     1
                    par_dict['prompt']=     10

                    for key in par_dict:
                        util.replace(key, str(par_dict[key]), info_file_name)

                    subprocess.call("./bin/CUFCM", shell=True)
                    util.savefile(par_dict, save_directory, 2)

                    sim_dict = util.read_scalar(save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat")
                    print("(", str(par_dict['alpha']), par_dict['beta'], par_dict['eta'], ") "
                          "Verror=", str(sim_dict["Verror"]),\
                          "Werror=", str(sim_dict["Werror"]),\
                          "time_compute=", str(sim_dict["time_compute"]))

                if(sys.argv[1] == 'read'):
                    sim_dict = util.read_scalar(save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat")
                    print(sim_dict)

    if(sys.argv[1] == 'run'):
        print("Data files moved to " + save_directory)

run()

# Parameter explanation:
# Alpha:    Sigma/dx
# Beta:     (ngd*dx)/Sigma
# Eta:      Rc/Sigma



























#