#!/bin/python3
from mpi4py import MPI
from mpmath import mp, mpf

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
# The first processor is leader, so one fewer available to be a worker
nworkers = nproc - 1
# samples
N = 100000000
DELTA = mpf(1.0) / N  # used mpf to calculate a precise value of pi
# integral
I = mpf(0.0)  # maintained this method throughout the calculation.
mp.dps = 14  # I have initially set the d.p.=14 to quickly test this 
#version works, I will gradually increase it as I alter my code to keep it in quick parallel.

def integrand(x):
# Added a docustring to explain the purpose, input and output of the function
#Technically it should return either no value or a worker will find the next value of pi.
    """
    Function to solve pi.

    Args: 
        x: Works out the value of pi, `mpf()` helps with accuracy and speeds this up.
        DELTA: Carries out the integration.

    Returns:
        None / DELTA
    """
    return mpf(4.0) / (mpf(1.0) + x * x)

if comm.Get_rank() == 0:  # Leader: choose points to sample function, send to workers and
    # collect their contributions. Also calculate a sub-set of points.
    for i in range(0, N):
        # decide which rank evaluates this point
        j = i % nproc
        # mid-point rule
        x = (i + 0.5) * DELTA
        if j == 0:
            # so do this locally using the leader machine
            y = integrand(x) * DELTA
        else:
            # communicate to a worker
            comm.send(x, dest=j)
            y = comm.recv(source=j)
        I += y

    # Shut down the workers
    for i in range(1, nproc):
        comm.send(-1.0, dest=i)

    # Only rank 0 should print the result
    print(f"The value of pi to 15 s.f. = {I}")

else:
    # Worker: waiting for something to happen, then stop if sent message
    # outside the integral limits
    while True:
        recv_x = comm.recv(source=0)
        if recv_x < 0.0:
            # stop the worker
            break
        comm.send(integrand(recv_x) * DELTA, dest=0)
# I have corrected the indentations here too.
