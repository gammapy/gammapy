# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Multiprocessing and multithreading setup"""

MULTIPROCESSING_BACKEND = "multiprocessing"
N_PROCESSES = 1
N_THREADS = 1

def get_multiprocessing(backend=None):
    if backend is None:
       backend = MULTIPROCESSING_BACKEND
    if backend == "multiprocessing":
        import multiprocessing
        return multiprocessing
    elif backend == "ray":
        import ray.util.multiprocessing as multiprocessing
        return multiprocessing
    else:
        raise ValueError("Invalid multiprocessing backend")
        