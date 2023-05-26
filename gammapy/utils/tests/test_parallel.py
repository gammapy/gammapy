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


def test_change_n_process_default():
    parallel.N_JOBS_DEFAULT = 5

    multiprocessing = parallel.get_multiprocessing()
    assert multiprocessing.__name__ == "multiprocessing"


@requires_dependency("ray")
def test_get_multiprocessing_ray():
    assert parallel.is_ray_available()
    assert not parallel.is_ray_initialized()

    multiprocessing = parallel.get_multiprocessing_ray()
    assert multiprocessing.__name__ == "ray.util.multiprocessing"


def test_run_multiprocessing_wrong_method():
    def func(arg):
        return arg

    with pytest.raises(ValueError):
        parallel.run_multiprocessing(
            func, [True, True], method="wrong_name", pool_kwargs=dict(processes=2)
        )
