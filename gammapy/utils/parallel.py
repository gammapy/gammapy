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


def run_multiprocessing(
    func, inputs, backend=None, pool_kwargs=None, method="starmap", method_kwargs=None
):
    multiprocessing = get_multiprocessing(backend)
    if method_kwargs is None:
        method_kwargs = {}
    if pool_kwargs is None:
        pool_kwargs = {}
    pool_kwargs.setdefault("processes", N_PROCESSES)
    if backend == "ray":
        pool_kwargs.setdefault("ray_adress", "auto")

    with multiprocessing.Pool(**pool_kwargs) as pool:
        if method == "starmap":
            return pool.starmap(func, progress_bar(inputs), **method_kwargs)
        elif method == "apply_async":
            results = []
            for arguments in progress_bar(inputs):
                result = pool.apply_async(func, arguments, **method_kwargs)
                results.append(result)
            # wait async run is done
            [result.wait() for result in results]
            return results
