# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Multiprocessing and multithreading setup"""
import importlib
import logging
import multiprocessing
from enum import Enum
from gammapy.utils.pbar import progress_bar

log = logging.getLogger(__name__)


class ParallelBackendEnum(Enum):
    """Enum for parallel backend"""

    multiprocessing = "multiprocessing"
    ray = "ray"

    @classmethod
    def from_str(cls, value):
        """Get enum from string"""
        if value is None:
            value = BACKEND_DEFAULT

        if value == "ray" and not is_ray_available():
            log.warning("Ray is not installed, falling back to multiprocessing backend")
            value = "multiprocessing"

        return cls(value)


class PoolMethodEnum(Enum):
    """Enum for pool method"""

    starmap = "starmap"
    apply_async = "apply_async"


BACKEND_DEFAULT = ParallelBackendEnum.multiprocessing
N_JOBS_DEFAULT = 1


def get_multiprocessing_ray():
    """Get multiprocessing module for ray backend"""
    import ray.util.multiprocessing as multiprocessing

    log.warning(
        "Gammapy support for parallelisation with ray is still a prototype and is not fully functional."
    )
    return multiprocessing


def is_ray_initialized():
    """Check if ray is initialized"""
    try:
        from ray import is_initialized

        return is_initialized()
    except ModuleNotFoundError:
        return False


def is_ray_available():
    """Check if ray is available"""
    try:
        importlib.import_module("ray")
        return True
    except ModuleNotFoundError:
        return False


class ParallelMixin:
    """Mixin class to handle parallel processing"""

    @property
    def n_jobs(self):
        """Number of jobs (int)"""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        """Number of jobs setter (int)"""
        if value is None:
            value = N_JOBS_DEFAULT

        self._n_jobs = int(value)

    @property
    def parallel_backend(self):
        """Parallel backend (str)"""
        return self._parallel_backend

    @parallel_backend.setter
    def parallel_backend(self, value):
        """Parallel backend setter (str)"""
        self._parallel_backend = ParallelBackendEnum.from_str(value).value


def run_multiprocessing(
    func,
    inputs,
    backend=BACKEND_DEFAULT,
    pool_kwargs=None,
    method="starmap",
    method_kwargs=None,
    task_name="",
):
    """Run function in a loop or in Parallel

    Parameters
    ----------
    func : function
        Function to run
    inputs : list
        List of arguments to pass to the function
    backend : {'multiprocessing', 'ray'}
        Backend to use.
    pool_kwargs : dict
        Keyword arguments passed to the pool
    method : {'starmap', 'apply_async'}
        Pool method to use.
    method_kwargs : dict
        Keyword arguments passed to the method
    task_name : str
        Name of the task to display in the progress bar
    """
    backend = ParallelBackendEnum.from_str(backend)

    if method_kwargs is None:
        method_kwargs = {}

    if pool_kwargs is None:
        pool_kwargs = {}

    processes = pool_kwargs.get("processes", N_JOBS_DEFAULT)

    multiprocessing = PARALLEL_BACKEND_MODULES[backend]

    if backend == ParallelBackendEnum.multiprocessing:
        if multiprocessing.current_process().name != "MainProcess":
            # subprocesses cannot have childs
            processes = 1
    # TODO: check for ray

    if processes == 1:
        return run_loop(
            func=func, inputs=inputs, method_kwargs=method_kwargs, task_name=task_name
        )

    if backend == ParallelBackendEnum.ray:
        address = "auto" if is_ray_initialized() else None
        pool_kwargs.setdefault("ray_address", address)

    log.info(f"Using {processes} processes to compute {task_name}")

    with multiprocessing.Pool(**pool_kwargs) as pool:
        pool_func = POOL_METHODS[PoolMethodEnum(method)]
        results = pool_func(
            pool=pool,
            func=func,
            inputs=inputs,
            method_kwargs=method_kwargs,
            task_name=task_name,
        )

    return results


def run_loop(func, inputs, method_kwargs=None, task_name=""):
    """Loop over inputs an run function"""
    results = []

    callback = method_kwargs.get("callback", None)

    for arguments in progress_bar(inputs, desc=task_name):
        result = func(*arguments)

        if callback is not None:
            result = callback(result)

        results.append(result)

    return results


def run_pool_star_map(pool, func, inputs, method_kwargs=None, task_name=""):
    """Run function in parallel"""
    return pool.starmap(func, progress_bar(inputs, desc=task_name), **method_kwargs)


def run_pool_async(pool, func, inputs, method_kwargs=None, task_name=""):
    """Run function in parallel async"""
    results = []

    for arguments in progress_bar(inputs, desc=task_name):
        result = pool.apply_async(func, arguments, **method_kwargs)
        results.append(result)

    # wait async run is done
    [result.wait() for result in results]
    return results


POOL_METHODS = {
    PoolMethodEnum.starmap: run_pool_star_map,
    PoolMethodEnum.apply_async: run_pool_async,
}

PARALLEL_BACKEND_MODULES = {
    ParallelBackendEnum.multiprocessing: multiprocessing,
}

if is_ray_available():
    PARALLEL_BACKEND_MODULES[ParallelBackendEnum.ray] = get_multiprocessing_ray()
