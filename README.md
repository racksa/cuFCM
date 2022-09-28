## DESCRIPTION

A CUDA benchmark for Fast Force-Coupling method

## USAGE
For Imperial College Maths Department users, on Nvidia4, in command line, run

```bash
nvidia-smi
```
to check available nodes, and then type

```bash
export CUDA_VISIBLE_DEVICES=x
```

to select the available node. 

Compile and run with

```bash
make clean CUFCM
./bin/CUFCM
```

## PYTHON SCRIPT
A Python script is provided to automatically run sequential simulations using a single binary file. This is achieved by replacing the text in a config file which is then read by the binary file.

To use the script, first modify the parameters in file script.py. The date generation and data reading/processing are separate process, and can be controled by system arguments passed in the terminal.

# Generate data
Run simulation with
```bash
python3 script.py run
```

# Post-process data
```bash
python3 script.py plot3 \#mode \#tol
```
Different
\#mode=1: plot error heatmap
\#mode=2: plot compute time
For \#mode=2, an additional argument \#tol can be passed to filter simulations of the specitied tol.