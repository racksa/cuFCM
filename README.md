## DESCRIPTION

A CUDA benchmark for the Fast force-coupling method

## COMPILE
Modify the `/src/config.hpp` file to select available options. To compile, under the home directory of this project, run
```bash
make clean x
```
where x is one of the given options in the makefile. The default path of the generated executables are under the repository `/bin/` .

## USAGE
For Imperial College Maths Department users, on the nvidia4 machine, run

```bash
nvidia-smi
```
to check node status, and then type

```bash
export CUDA_VISIBLE_DEVICES=x
```

to select the available node. 

Run with

```bash
./bin/x
```
where x is the name of the executable binary.

## PYTHON SCRIPT
A Python script is provided to automatically run sequential simulations using a single binary file. This is achieved by replacing the text in a config file which is then read by the binary file.

> :warning: **The Python script is very custom written and does not run out of the box**: Be very careful here!

To use that, first change the path in 'settings.py' to match your fast fcm directory path. Create the required directory for data saving. 

You will need to use the random generator by compiling
''' make RANDOM_GENERATOR
'''

To use the script, modify the parameters in file `script.py`. The member function `start_loop` can be modified to sweep the simulation parameters. Data generation and data reading/processing are separate process, and can be controled by system arguments passed in the terminal.


## GENERATING DATA
Run simulations with
```bash
python3 script.py run
```