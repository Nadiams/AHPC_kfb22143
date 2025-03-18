#!/usr/bin/env
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 12:20:28 2025

@author: nadia
"""

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
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=00:40:00
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
pylint --extension-pkg-whitelist=mpi4py.MPI assignment_3_3.py

#======================================================
# Prologue script to record job details
# Do not change the line below
#======1================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

# Modify the line below to run your program
mpirun -np $SLURM_NPROCS python3 assignment_3_3.py

# Run pylint to check for formatting issues
echo "Running pylint for code quality check..."
pylint --extension-pkg-whitelist=mpi4py.MPI assignment_3_3.py

# If pylint exits with an error, print a message but continue
if [ $? -ne 0 ]; then
    echo "Pylint found issues, but continuing with execution..."
fi

# Run the MPI Python program
echo "Starting MPI computation..."
#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
