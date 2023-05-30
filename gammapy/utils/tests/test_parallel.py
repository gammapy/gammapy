# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import gammapy.utils.parallel as parallel
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


@pytest.mark.parametrize("method", ["starmap", "apply_async"])
def test_run_multiprocessing_simple(method):
    1 / 0

    def square(x):
        return x**2

    N = 10
    inputs = range(N + 1)

    result = parallel.run_multiprocessing(
        func=square,
        inputs=inputs,
        methode=method,
        pool_kwargs=dict(processes=2),
    )
    assert sum(result) == N * (N + 1) * (2 * N + 1) / 6


@requires_dependency("ray")
@pytest.mark.parametrize("method", ["starmap", "apply_async"])
def test_run_multiprocessing_simple_ray(method):
    def square(x):
        return x**2

    N = 10
    inputs = range(N + 1)

    result = parallel.run_multiprocessing(
        func=square,
        inputs=inputs,
        methode=method,
        pool_kwargs=dict(processes=2),
        parallel_backend="ray",
    )
    assert sum(result) == N * (N + 1) * (2 * N + 1) / 6
