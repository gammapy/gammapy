# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import astropy.units as u
import gammapy.utils.parallel as parallel
from gammapy.estimators import FluxPointsEstimator
from gammapy.utils.testing import requires_dependency


def test_parallel_mixin():
    p = parallel.ParallelMixin()

    with pytest.raises(ValueError):
        p.parallel_backend = "wrong_name"

    with pytest.raises(ValueError):
        p.n_jobs = "5 jobs"


@requires_dependency("ray")
def test_get_multiprocessing_ray():
    assert parallel.is_ray_available()

    multiprocessing = parallel.get_multiprocessing_ray()
    assert multiprocessing.__name__ == "ray.util.multiprocessing"


def test_run_multiprocessing_wrong_method():
    def func(arg):
        return arg

    with pytest.raises(ValueError):
        parallel.run_multiprocessing(
            func, [True, True], method="wrong_name", pool_kwargs=dict(processes=2)
        )


def square(x):
    return x**2


class MyTask:
    def __init__(self):
        self.sum_squared = 0

    def __call__(self, x):
        return x**2

    def callback(self, result):
        self.sum_squared += result


def test_run_multiprocessing_simple_starmap():
    N = 10
    inputs = [(_,) for _ in range(N + 1)]

    result = parallel.run_multiprocessing(
        func=square,
        inputs=inputs,
        method="starmap",
        pool_kwargs=dict(processes=2),
    )
    assert sum(result) == N * (N + 1) * (2 * N + 1) / 6


def test_run_multiprocessing_simple_apply_async():
    N = 10
    inputs = [(_,) for _ in range(N + 1)]

    task = MyTask()

    _ = parallel.run_multiprocessing(
        func=task,
        inputs=inputs,
        method="apply_async",
        pool_kwargs=dict(processes=2),
        method_kwargs=dict(callback=task.callback),
    )
    assert task.sum_squared == N * (N + 1) * (2 * N + 1) / 6


@requires_dependency("ray")
def test_run_multiprocessing_simple_ray_starmap():
    N = 10
    inputs = [(_,) for _ in range(N + 1)]

    result = parallel.run_multiprocessing(
        func=square,
        inputs=inputs,
        method="starmap",
        pool_kwargs=dict(processes=2),
        backend="ray",
    )
    assert sum(result) == N * (N + 1) * (2 * N + 1) / 6


@requires_dependency("ray")
def test_run_multiprocessing_simple_ray_apply_async():
    N = 10
    inputs = [(_,) for _ in range(N + 1)]

    task = MyTask()

    _ = parallel.run_multiprocessing(
        func=task,
        inputs=inputs,
        method="apply_async",
        pool_kwargs=dict(processes=2),
        method_kwargs=dict(callback=task.callback),
        backend="ray",
    )
    assert task.sum_squared == N * (N + 1) * (2 * N + 1) / 6


@requires_dependency("ray")
def test_multiprocessing_manager():
    with parallel.multiprocessing_manager(
        backend="ray",
        method="apply_async",
        pool_kwargs=dict(processes=2),
        method_kwargs=dict(callback=None),
    ):
        assert parallel.BACKEND_DEFAULT == "ray"
        assert parallel.POOL_KWARGS_DEFAULT["processes"] == 2
        assert parallel.METHOD_DEFAULT == "apply_async"
        assert "callback" in parallel.METHOD_KWARGS_DEFAULT

    assert parallel.BACKEND_DEFAULT == parallel.ParallelBackendEnum.multiprocessing
    assert parallel.POOL_KWARGS_DEFAULT["processes"] == 1
    assert parallel.METHOD_DEFAULT == parallel.PoolMethodEnum.starmap
    assert "callback" not in parallel.METHOD_KWARGS_DEFAULT

    fpe = FluxPointsEstimator(energy_edges=[1, 3, 10] * u.TeV)
    assert fpe.parallel_backend == parallel.ParallelBackendEnum.multiprocessing

    with parallel.multiprocessing_manager(backend="ray", pool_kwargs=dict(processes=2)):
        assert fpe.parallel_backend == "ray"
        assert fpe.n_jobs == 2

        fpe = FluxPointsEstimator(energy_edges=[1, 3, 10] * u.TeV)
        assert fpe.parallel_backend == "ray"
    assert fpe.parallel_backend == parallel.ParallelBackendEnum.multiprocessing

    fpe = FluxPointsEstimator(
        energy_edges=[1, 3, 10] * u.TeV, n_jobs=2, parallel_backend="multiprocessing"
    )
    with parallel.multiprocessing_manager(backend="ray", pool_kwargs=dict(processes=3)):
        assert fpe.parallel_backend == "multiprocessing"
        assert fpe.n_jobs == 2
