import sys
import subprocess
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
                    npts = 256

                    par_dict['N']=          500000.0
                    par_dict['rh']=         0.02609300415934458
                    par_dict['alpha']=      0.9352
                    par_dict['beta']=       9.706
                    par_dict['eta']=        5.573
                    par_dict['nx']=         npts
                    par_dict['ny']=         npts
                    par_dict['nz']=         npts
                    par_dict['repeat']=     1

                    for key in par_dict:
                        util.replace(key, str(par_dict[key]), info_file_name)

                    subprocess.call("./bin/CUFCM", shell=True)
                    util.savefile(par_dict, save_directory, 2)
                    print("Data files moved to " + save_directory)

                if(sys.argv[1] == 'read'):
                    sim_dict = util.read_scalar(save_directory + "simulation_scalar" + util.parser(par_dict) + ".dat")
                    print(sim_dict)

run()

# Parameter explanation:
# Alpha:    Sigma/dx
# Beta:     (ngd*dx)/Sigma
# Eta:      Rc/Sigma



























#