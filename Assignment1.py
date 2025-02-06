#!/bin/python3
"""
	Sets environment for program to work in, including the conda environment: condampipython.
	Args:
		python3
	Returns:
		mpmath, mpi4py, math.fsum
"""
from mpi4py import MPI
from mpmath import mp, mpf
#from math.fsum import fsum

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
# The first processor is leader, so one fewer available to be a worker
nworkers = nproc - 1
# samples
N = 100000000
DELTA = mpf(1.0) / N
# used mpf to calculate a precise value of pi
# integral
I = mpf(0.0)
mp.dps = 50
# I have initially set the d.p.=2 to quickly test this version works,
#I will gradually increase it as I alter my code to keep it in quick parallel.
#Have now moved to 14 d.p.

def integrand(x):
    """
    Function to solve pi.
    Args:
        x (mpf): x is evaluated by the workers and fed back through a 
        loop in parallel with each other.
    Returns:
        mpf: the value of pi worked out by the integrand function.
    """
    return mpf(4.0) / (mpf(1.0) + x * x)
    # maintained this method throughout the calculation.
if comm.Get_rank() == 0: # Leader: choose points to sample function, send to workers and
    # collect their contributions. Also calculate a sub-set of points.
    for i in range(0, N):
        # decide which rank evaluates this point
        j = i % nproc
        # mid-point rule
        recv_x = (i + 0.5) * DELTA
        if j == 0:
            # so do this locally using the leader machine
            y = integrand(recv_x) * DELTA
        else:
            # communicate to a worker
            comm.send(recv_x, dest=j)
            y = comm.recv(source=j)
        I += comm.reduce(y, op=MPI.SUM, root=0)

    # Shut down the workers
    for i in range(1, nproc):
        workersection = integrand(recv_x) * DELTA # Calculate partial sum in each worker
        comm.send(workersection, dest=0)
    print(f"The value of pi to 15 s.f. = {float(I):.14f}")

else:
    # Worker: waiting for something to happen, then stop if sent message
    # outside the integral limits
    workersection = mpf(0.0)
    while True:
        recv_x = comm.recv(source=0)
        if recv_x < 0.0:
            # stop the worker
            break
        workersection += integrand(recv_x) * DELTA
    comm.send(integrand(recv_x) * DELTA, dest=0)