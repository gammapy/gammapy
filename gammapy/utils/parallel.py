# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Multiprocessing and multithreading setup."""

import importlib
import logging
from enum import Enum
from gammapy.utils.pbar import progress_bar

log = logging.getLogger(__name__)

__all__ = [
    "multiprocessing_manager",
    "run_multiprocessing",
    "BACKEND_DEFAULT",
    "N_JOBS_DEFAULT",
    "POOL_KWARGS_DEFAULT",
    "METHOD_DEFAULT",
    "METHOD_KWARGS_DEFAULT",
    "convolve_psf_gpu",
    "get_gpu_device",
]


class ParallelBackendEnum(Enum):
    """Enum for parallel backend."""

    multiprocessing = "multiprocessing"
    ray = "ray"

    @classmethod
    def from_str(cls, value):
        """Get enum from string."""
        if value == "ray" and not is_ray_available():
            log.warning("Ray is not installed, falling back to multiprocessing backend")
            value = "multiprocessing"

        return cls(value)


class PoolMethodEnum(Enum):
    """Enum for pool method."""

    starmap = "starmap"
    apply_async = "apply_async"


BACKEND_DEFAULT = ParallelBackendEnum.multiprocessing
N_JOBS_DEFAULT = 1
ALLOW_CHILD_JOBS = False
POOL_KWARGS_DEFAULT = dict(processes=N_JOBS_DEFAULT)
METHOD_DEFAULT = PoolMethodEnum.starmap
METHOD_KWARGS_DEFAULT = {}


def get_multiprocessing():
    """Get multiprocessing module."""
    import multiprocessing

    return multiprocessing


def get_multiprocessing_ray():
    """Get multiprocessing module for ray backend."""
    import ray.util.multiprocessing as multiprocessing

    log.warning(
        "Gammapy support for parallelisation with ray is still a prototype and is not fully functional."
    )
    return multiprocessing


def is_ray_initialized():
    """Check if ray is initialized."""
    try:
        from ray import is_initialized

        return is_initialized()
    except ModuleNotFoundError:
        return False


def is_ray_available():
    """Check if ray is available."""
    try:
        importlib.import_module("ray")
        return True
    except ModuleNotFoundError:
        return False


class multiprocessing_manager:
    """Context manager to update the default configuration for multiprocessing.

    Only the default configuration will be modified, if class arguments like
    `n_jobs` and `parallel_backend` are set they will overwrite the default configuration.

    Parameters
    ----------
    backend : {'multiprocessing', 'ray'}
        Backend to use.
    pool_kwargs : dict
        Keyword arguments passed to the pool. The number of processes is limited
        to the number of physical CPUs.
    method : {'starmap', 'apply_async'}
        Pool method to use.
    method_kwargs : dict
        Keyword arguments passed to the method

    Examples
    --------
    ::

        import gammapy.utils.parallel as parallel
        from gammapy.estimators import FluxPointsEstimator

        fpe = FluxPointsEstimator(energy_edges=[1, 3, 10] * u.TeV)

        with parallel.multiprocessing_manager(
                backend="multiprocessing",
                pool_kwargs=dict(processes=2),
            ):
            fpe.run(datasets)
    """

    def __init__(self, backend=None, pool_kwargs=None, method=None, method_kwargs=None):
        global \
            BACKEND_DEFAULT, \
            POOL_KWARGS_DEFAULT, \
            METHOD_DEFAULT, \
            METHOD_KWARGS_DEFAULT, \
            N_JOBS_DEFAULT
        self._backend = BACKEND_DEFAULT
        self._pool_kwargs = POOL_KWARGS_DEFAULT
        self._method = METHOD_DEFAULT
        self._method_kwargs = METHOD_KWARGS_DEFAULT
        self._n_jobs = N_JOBS_DEFAULT
        if backend is not None:
            BACKEND_DEFAULT = ParallelBackendEnum.from_str(backend).value
        if pool_kwargs is not None:
            POOL_KWARGS_DEFAULT = pool_kwargs
            N_JOBS_DEFAULT = pool_kwargs.get("processes", N_JOBS_DEFAULT)
        if method is not None:
            METHOD_DEFAULT = PoolMethodEnum(method).value
        if method_kwargs is not None:
            METHOD_KWARGS_DEFAULT = method_kwargs

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        global \
            BACKEND_DEFAULT, \
            POOL_KWARGS_DEFAULT, \
            METHOD_DEFAULT, \
            METHOD_KWARGS_DEFAULT, \
            N_JOBS_DEFAULT
        BACKEND_DEFAULT = self._backend
        POOL_KWARGS_DEFAULT = self._pool_kwargs
        METHOD_DEFAULT = self._method
        METHOD_KWARGS_DEFAULT = self._method_kwargs
        N_JOBS_DEFAULT = self._n_jobs


class ParallelMixin:
    """Mixin class to handle parallel processing."""

    _n_child_jobs = 1

    @property
    def n_jobs(self):
        """Number of jobs as an integer."""
        # TODO: this is somewhat unusual behaviour. It deviates from a normal default value handling
        if self._n_jobs is None:
            return N_JOBS_DEFAULT

        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        """Number of jobs setter as an integer."""
        if not isinstance(value, (int, type(None))):
            raise ValueError(
                f"Invalid type: {value!r}, and integer or None is expected."
            )

        self._n_jobs = value
        if ALLOW_CHILD_JOBS:
            self._n_child_jobs = value

    def _update_child_jobs(self):
        """needed because we can update only in the main process
        otherwise global ALLOW_CHILD_JOBS has default value"""
        if ALLOW_CHILD_JOBS:
            self._n_child_jobs = self.n_jobs
        else:
            self._n_child_jobs = 1

    @property
    def _get_n_child_jobs(self):
        """Number of allowed child jobs as an integer."""
        return self._n_child_jobs

    @property
    def parallel_backend(self):
        """Parallel backend as a string."""
        if self._parallel_backend is None:
            return BACKEND_DEFAULT

        return self._parallel_backend

    @parallel_backend.setter
    def parallel_backend(self, value):
        """Parallel backend setter (str)"""
        if value is None:
            self._parallel_backend = None
        else:
            self._parallel_backend = ParallelBackendEnum.from_str(value).value


def run_multiprocessing(
    func,
    inputs,
    backend=None,
    pool_kwargs=None,
    method=None,
    method_kwargs=None,
    task_name="",
):
    """Run function in a loop or in Parallel.

    Notes
    -----
    The progress bar can be displayed for this function.

    Parameters
    ----------
    func : function
        Function to run.
    inputs : list
        List of arguments to pass to the function.
    backend : {'multiprocessing', 'ray'}, optional
        Backend to use. Default is None.
    pool_kwargs : dict, optional
        Keyword arguments passed to the pool. The number of processes is limited
        to the number of physical CPUs. Default is None.
    method : {'starmap', 'apply_async'}
        Pool method to use. Default is "starmap".
    method_kwargs : dict, optional
        Keyword arguments passed to the method. Default is None.
    task_name : str, optional
        Name of the task to display in the progress bar. Default is "".
    """
    if backend is None:
        backend = BACKEND_DEFAULT

    if method is None:
        method = METHOD_DEFAULT

    if method_kwargs is None:
        method_kwargs = METHOD_KWARGS_DEFAULT

    if pool_kwargs is None:
        pool_kwargs = POOL_KWARGS_DEFAULT

    try:
        method_enum = PoolMethodEnum(method)
    except ValueError as e:
        raise ValueError(f"Invalid method: {method}") from e

    processes = pool_kwargs.get("processes", N_JOBS_DEFAULT)

    backend = ParallelBackendEnum.from_str(backend)
    multiprocessing = PARALLEL_BACKEND_MODULES[backend]()

    if backend == ParallelBackendEnum.multiprocessing:
        cpu_count = multiprocessing.cpu_count()

        if processes > cpu_count:
            log.info(f"Limiting number of processes from {processes} to {cpu_count}")
            processes = cpu_count

        if multiprocessing.current_process().name != "MainProcess":
            # with multiprocessing subprocesses cannot have children (but possible with ray)
            processes = 1

    if processes == 1:
        return run_loop(
            func=func, inputs=inputs, method_kwargs=method_kwargs, task_name=task_name
        )

    if backend == ParallelBackendEnum.ray:
        address = "auto" if is_ray_initialized() else None
        pool_kwargs.setdefault("ray_address", address)

    log.info(f"Using {processes} processes to compute {task_name}")

    with multiprocessing.Pool(**pool_kwargs) as pool:
        pool_func = POOL_METHODS[method_enum]
        results = pool_func(
            pool=pool,
            func=func,
            inputs=inputs,
            method_kwargs=method_kwargs,
            task_name=task_name,
        )

    return results


def run_loop(func, inputs, method_kwargs=None, task_name=""):
    """Loop over inputs and run function."""
    results = []

    callback = method_kwargs.get("callback", None)

    for arguments in progress_bar(inputs, desc=task_name):
        result = func(*arguments)

        if callback is not None:
            result = callback(result)

        results.append(result)

    return results


def run_pool_star_map(pool, func, inputs, method_kwargs=None, task_name=""):
    """Run function in parallel."""
    return pool.starmap(func, progress_bar(inputs, desc=task_name), **method_kwargs)


def run_pool_async(pool, func, inputs, method_kwargs=None, task_name=""):
    """Run function in parallel async."""
    results = []

    for arguments in progress_bar(inputs, desc=task_name):
        result = pool.apply_async(func, arguments, **method_kwargs)
        results.append(result)
    # wait async run is done
    [result.wait() for result in results]
    return results


def is_cuda_available():
    """Check whether CUDA is available via torch."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_device():
    """Get default GPU device.

    Returns
    -------
    device : torch.device or None
        CUDA device if available, otherwise None.
    """
    if not is_cuda_available():
        return None

    import torch

    return torch.device("cuda")


# ============================
# GPU convolution helper
# ============================


def _convolve2d_grouped_gpu(x_np, k_np, device):
    """Low-level grouped 2D convolution on GPU (torch).

    Parameters
    ----------
    x_np : numpy.ndarray
        Input array with shape (Y, X) or (E, Y, X).
    k_np : numpy.ndarray
        Kernel array with shape (Ky, Kx) or (Ek, Ky, Kx).
    device : torch.device
        Torch device (e.g. torch.device("cuda")).

    Returns
    -------
    y_np : numpy.ndarray
        Output array with shape (E, Y, X). If input was (Y, X), E=1.
    """
    import torch
    import torch.nn.functional as F

    # normalize shapes to (E,Y,X) and (Ek,Ky,Kx)
    x_was_2d = x_np.ndim == 2
    if x_was_2d:
        x_np = x_np[None, ...]  # (1, Y, X)

    if k_np.ndim == 2:
        k_np = k_np[None, ...]  # (1, Ky, Kx)

    if x_np.ndim != 3 or k_np.ndim != 3:
        raise ValueError(
            f"Expected x_np and k_np to be 3D, got {x_np.ndim=} {k_np.ndim=}"
        )

    E, Y, X = x_np.shape
    Ek, Ky, Kx = k_np.shape

    # single transfer
    x = torch.as_tensor(x_np, device=device)
    if not x.is_floating_point():
        x = x.float()

    k = torch.as_tensor(k_np, device=device, dtype=x.dtype)

    # Build per-plane kernels: (E, Ky, Kx)
    if Ek == 1:
        k_full = k.expand(E, -1, -1)  # same kernel for all planes
    elif Ek == E:
        k_full = k
    else:
        # match old behavior: ke = min(e, Ek-1)
        ke = torch.clamp(torch.arange(E, device=device), max=Ek - 1)
        k_full = k.index_select(0, ke)

    # conv2d does correlation -> flip once to get convolution
    k_full = torch.flip(k_full, dims=(-2, -1))
    weight = k_full[:, None, :, :]  # (E,1,Ky,Kx)

    # Grouped conv: input (N=1, C=E, H=Y, W=X), groups=E
    x4 = x[None, :, :, :]  # (1,E,Y,X)

    pad_y = Ky // 2
    pad_x = Kx // 2

    with torch.inference_mode():
        y4 = F.conv2d(
            x4,
            weight,
            bias=None,
            stride=1,
            padding=(pad_y, pad_x),  # ZERO padding (matches "same" w/ zero extension)
            groups=E,
        )  # (1,E,Y,X)

    y = y4[0]  # (E,Y,X)
    y_np = y.detach().cpu().numpy()
    return y_np


def _convolve_spatial_gpu(x_tensor, k_tensor):
    """Spatial domain convolution using PyTorch (nn.functional.conv2d)"""
    import torch.nn.functional as F

    # x: (E, Y, X), k: (E, Ky, Kx)
    # conv2d requires input: (Batch, C, H, W) and weight: (Out_C, In_C, H, W)
    E, H, W = x_tensor.shape
    Ek, Ky, Kx = k_tensor.shape

    # Pad to maintain 'same' size
    pad_y, pad_x = Ky // 2, Kx // 2
    x_padded = F.pad(
        x_tensor.unsqueeze(0), (pad_x, pad_x, pad_y, pad_y), mode="constant", value=0
    )

    # Grouped Convolution: each energy bin uses its own PSF kernel
    # groups = E allows each channel to be convolved independently
    weight = k_tensor.unsqueeze(1)  # (E, 1, Ky, Kx)

    output = F.conv2d(x_padded, weight, groups=E)
    return output.squeeze(0)


def _convolve_fft_gpu(x_tensor, k_tensor):
    """Frequency domain convolution using PyTorch FFT"""
    import torch

    E, H, W = x_tensor.shape
    Ek, Ky, Kx = k_tensor.shape

    # Pad to H+Ky-1 to avoid circular convolution artifacts
    n_y = H + Ky - 1
    n_x = W + Kx - 1

    # 1. Transform to frequency domain
    X_f = torch.fft.rfftn(x_tensor, s=(n_y, n_x))
    K_f = torch.fft.rfftn(k_tensor, s=(n_y, n_x))

    # 2. Pointwise multiplication and inverse transform
    conv_full = torch.fft.irfftn(X_f * K_f, s=(n_y, n_x))

    # 3. Crop to 'same' size (centered alignment)
    start_y = Ky // 2
    start_x = Kx // 2
    return conv_full[:, start_y : start_y + H, start_x : start_x + W]


def convolve_psf_gpu(npred, psf, device):
    """
    GPU-optimized PSF convolution with automatic switching between FFT and Spatial methods.
    """
    import torch
    import numpy as np
    from gammapy.maps import Map

    x_np = npred.data.astype(np.float32)
    k_np = psf.psf_kernel_map.data.astype(np.float32)

    # Handle broadcasting (e.g., 2D image with a 3D PSF cube)
    if x_np.ndim == 2 and k_np.ndim == 3:
        geom_out = npred.geom.to_image().to_cube(axes=psf.psf_kernel_map.geom.axes)
        Ek = k_np.shape[0]
        x_np = np.broadcast_to(x_np, (Ek,) + x_np.shape).copy()
        x_was_2d = False
    else:
        geom_out = npred.geom
        x_was_2d = x_np.ndim == 2
        if x_was_2d:
            x_np = x_np[np.newaxis, ...]
            k_np = k_np[np.newaxis, ...]

    # Transfer data to GPU
    x_tensor = torch.from_numpy(x_np).to(device)
    k_tensor = torch.from_numpy(k_np).to(device)

    # Selection logic: Use FFT if the kernel size exceeds 31x31
    kernel_size = max(k_tensor.shape[-2:])

    if kernel_size > 31:
        y_tensor = _convolve_fft_gpu(x_tensor, k_tensor)
    else:
        y_tensor = _convolve_spatial_gpu(x_tensor, k_tensor)

    y_np = y_tensor.cpu().numpy()

    if x_was_2d:
        y_np = y_np[0]

    return Map.from_geom(geom_out, data=y_np, unit=npred.unit)


POOL_METHODS = {
    PoolMethodEnum.starmap: run_pool_star_map,
    PoolMethodEnum.apply_async: run_pool_async,
}

PARALLEL_BACKEND_MODULES = {
    ParallelBackendEnum.multiprocessing: get_multiprocessing,
    ParallelBackendEnum.ray: get_multiprocessing_ray,
}
