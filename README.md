# Building and running instructions

## CUDA
### Locally:
Install [Nvidia's CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and make sure `nvcc` is in your `PATH`.\
Build with: `make`.\
Run, e.g., with: `./main 128 128 64 2 1024 laplap-global`.\
Clean with: `make clean`.

### On Piz Daint:
Make sure `PrgEnv-nvidia` is loaded.\
Build with: `make`.\
Run, e.g., with: `srun -A class03 -C gpu ./main 128 128 64 2 1024 laplap-global`.\
Clean with: `make clean`.

## OpenACC
### Locally:
Install [Nvidia's HPC SDK](https://developer.nvidia.com/hpc-sdk) and make sure `nvc++` is in your `PATH`.\
Build with: `make`.\
Run, e.g., with: `./main 128 128 64 2 1024 parallel`.\
Clean with: `make clean`.

### On Piz Daint:
Make sure `PrgEnv-nvidia` is loaded.\
Build with: `make`.\
Run, e.g., with: `srun -A class03 -C gpu ./main 128 128 64 2 1024 parallel`.\
Clean with: `make clean`.

## CuPy
### Locally:
Create a new virtual environment: `python -m venv .venv`.\
Activate the environment: `source .venv/bin/activate`.\
Install the dependencies: `pip install -r requirements.txt`.\
Run, e.g., with: `python main.py -nx 128 -ny 128 -nz 64 -bdry 2 -itrs 1024`.

### On Piz Daint:
Make sure you have the `HPC4WC_kernel` IPython kernel from the [HPC4WC setup script](https://github.com/ofuhrer/HPC4WC/blob/main/setup/HPC4WC_setup.sh) installed.\
Open `run.ipynb` via the [CSCS JupyterHub](https://jupyter.cscs.ch).\
Run a cell such as: `!python main.py -nx 128 -ny 128 -nz 64 -bdry 2 -itrs 1024`.

## GT4Py
### Locally:
_**Note:** GT4Py is currently incompatible with the latest version of Python (v3.12)._\
_Therefore, `python3.10` is used in the following._

Create a new virtual environment: `python3.10 -m venv .venv`.\
Activate the environment: `source .venv/bin/activate`.\
Install the dependencies: `pip install -r requirements.txt`.\
Run, e.g., with: `python xyz-laplap.py -nx 128 -ny 128 -nz 64 -bdry 2 -itrs 1024 -bknd cuda`.

### On Piz Daint:
Make sure you have the `HPC4WC_kernel` IPython kernel from the [HPC4WC setup script](https://github.com/ofuhrer/HPC4WC/blob/main/setup/HPC4WC_setup.sh) installed.\
Open `run.ipynb` via the [CSCS JupyterHub](https://jupyter.cscs.ch).\
Run a cell such as: `!python xyz-laplap.py -nx 128 -ny 128 -nz 64 -bdry 2 -itrs 1024 -bknd cuda`.
