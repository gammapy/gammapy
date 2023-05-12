# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Multiprocessing and multithreading setup"""
import logging
import os
from gammapy.utils.pbar import progress_bar

log = logging.getLogger(__name__)


MULTIPROCESSING_BACKEND = "multiprocessing"
N_PROCESSES = 1
N_THREADS = 1


class ParallelMixin:
    @property
    def n_jobs(self):
        if self._n_jobs is None:
            return N_PROCESSES
        else:
            return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = value

    @property
    def parallel_backend(self):
        if self._parallel_backend is None:
            return MULTIPROCESSING_BACKEND
        else:
            return self._parallel_backend

    @parallel_backend.setter
    def parallel_backend(self, value):
        self._parallel_backend = value


def get_multiprocessing(backend=None):
    """import multiprocessing module for a given backend"""

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
    """Run function in a loop or in Parallel"""

    if method not in ["starmap", "apply_async"]:
        raise ValueError("Invalid multiprocessing method")

    multiprocessing = get_multiprocessing(backend)
    if method_kwargs is None:
        method_kwargs = {}
    if pool_kwargs is None:
        pool_kwargs = {}
    pool_kwargs.setdefault("processes", N_PROCESSES)
    if backend == "ray":
        pool_kwargs.setdefault("ray_address", "auto")

    processes = pool_kwargs["processes"]
    if backend == "multiprocessing":
        processes = max(processes, os.cpu_count())
        if multiprocessing.current_process().name != "MainProcess":
            # subprocesses cannot have childs
            processes = 1
    # TODO: check for ray

    log.info(f"Using {processes} processes to compute {task_name}")

    if processes == 1:
        return run_loop(func, inputs, method_kwargs, task_name)
    else:
        with multiprocessing.Pool(**pool_kwargs) as pool:
            return run_pool(pool, func, inputs, method, method_kwargs, task_name)


def run_loop(func, inputs, method_kwargs=None, task_name=""):
    """Loop over inputs an run function"""

    results = []
    for arguments in progress_bar(inputs, desc=task_name):
        output = func(*arguments)
        if "callback" in method_kwargs:
            results.append(method_kwargs["callback"](output))
        else:
            results.append(output)
    return results


def run_pool(pool, func, inputs, method="starmap", method_kwargs=None, task_name=""):
    """Run function in Parallel"""

    if method == "starmap":
        return pool.starmap(func, progress_bar(inputs, desc=task_name), **method_kwargs)
    elif method == "apply_async":
        results = []
        for arguments in progress_bar(inputs, desc=task_name):
            result = pool.apply_async(func, arguments, **method_kwargs)
            results.append(result)
        # wait async run is done
        [result.wait() for result in results]
        return results
