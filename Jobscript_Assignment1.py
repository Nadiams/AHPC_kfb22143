#!/bin/bash

#======================================================
#
# Job script for running a serial job on a single core 
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching-gpu
#
# Specify project account
#SBATCH --account=teaching
#
# No. of tasks required (ntasks=1 for a single-core job)
#SBATCH --ntasks=16
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=01:40:00
#
# Job name
#SBATCH --job-name=pythontest
#
# Output file
#SBATCH --output=task1_16cores-%j.out
#======================================================

module purge

#Example module load command. 
#Load any modules appropriate for your program's requirements
#module load lammps/intel-2018.2/22Aug18
module load openmpi/gcc-8.5.0/4.1.1


#======================================================
# Prologue script to record job details
# Do not change the line below
#======1================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

# Modify the line below to run your program
mpirun -np $SLURM_NPROCS python3 finaltry3.py
pylint --extension-pkg-whitelist=mpi4py.MPI finaltry3.py
#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
