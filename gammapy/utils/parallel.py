# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Multiprocessing and multithreading setup"""
from gammapy.utils.pbar import progress_bar

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


def run_starmap(func, inputs, backend=None, pool_kwargs=None, starmap_kwargs=None):
    multiprocessing = get_multiprocessing(backend)
    if starmap_kwargs is None:
        starmap_kwargs = {}

    if pool_kwargs is None:
        pool_kwargs = {}
    pool_kwargs.setdefault("processes", N_PROCESSES)
    if backend == "ray":
        pool_kwargs.setdefault("ray_adress", "auto")

    with multiprocessing.Pool(**pool_kwargs) as pool:
        return pool.starmap(func, progress_bar(inputs), **starmap_kwargs)
