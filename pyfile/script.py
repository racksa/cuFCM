from distutils.log import error
import sys
from pathlib import Path
import os
import subprocess
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pylab import *
from settings import *
import util
import simulator

sim = simulator.SIM()

if(sys.argv[1] == 'run' or sys.argv[1] == 'read'):
    if(len(sys. argv)>2):
        if(sys.argv[2] == 'both'):
            for j in range(2):
                sim.mod_solver(j)
                sim.start_loop()
    else:
        sim.start_loop()

# deprecated
if(sys.argv[1] == 'test'):
    sim.run_test()

if(sys.argv[1] == 'single'):
    sim.run_single()

if(sys.argv[1] == 'two_particle'):
    sim.run_two_particle()

if(sys.argv[1] == 'read_npy'):
    sim.analyse()
    sim.plot_ptps_vs_phi()
    sim.plot_sigfac()

if(sys.argv[1] == 'plot_combine'):
    # date = "20221125"
    sim.analyse_and_plot_both()
    
if(sys.argv[1] == 'plot_piechart'):
    sim.analyse()
    sim.plot_pie_chart_of_time()

if(sys.argv[1] == 'flow_field'):
    sim.evaluate_flow_field()


# Parameter explanation:
# Alpha:    Sigma/dx
# Beta:     (ngd*dx)/Sigma
# Eta:      Rc/Sigma

# par_dict['alpha']=      0.9352
# par_dict['beta']=       9.706
# par_dict['eta']=        5.573




#