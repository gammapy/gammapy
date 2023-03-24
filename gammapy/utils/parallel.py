# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Multiprocessing and multithreading setup"""
import logging
from gammapy.utils.pbar import progress_bar

log = logging.getLogger(__name__)


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
    func,
    inputs,
    backend=None,
    pool_kwargs=None,
    method="starmap",
    method_kwargs=None,
    task_name="",
):
    if method not in ["starmap", "apply_async"]:
        raise ValueError("Invalid multiprocessing method")

    multiprocessing = get_multiprocessing(backend)
    if method_kwargs is None:
        method_kwargs = {}
    if pool_kwargs is None:
        pool_kwargs = {}
    pool_kwargs.setdefault("processes", N_PROCESSES)
    if backend == "ray":
        pool_kwargs.setdefault("ray_adress", "auto")

    processes = pool_kwargs["processes"]
    log.info(f"Using {processes} processes to compute {task_name}")

    if processes == 1:
        results = []
        for arguments in progress_bar(inputs, desc=task_name):
            output = func(*arguments)
            if method == "apply_async" and "callback" in method_kwargs:
                results.append(method_kwargs["callback"](output))
            else:
                results.append(output)
        return results
    else:
        with multiprocessing.Pool(**pool_kwargs) as pool:
            if method == "starmap":
                return pool.starmap(
                    func, progress_bar(inputs, desc=task_name), **method_kwargs
                )
            elif method == "apply_async":
                results = []
                for arguments in progress_bar(inputs, desc=task_name):
                    result = pool.apply_async(func, arguments, **method_kwargs)
                    results.append(result)
                # wait async run is done
                [result.wait() for result in results]
                return results
