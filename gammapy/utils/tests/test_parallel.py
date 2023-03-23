# Licensed under a 3-clause BSD style license - see LICENSE.rst
import gammapy.utils.parallel as parallel


def test_get_multiprocessing():
    parallel.MULTIPROCESSING_BACKEND = "multiprocessing"
    multiprocessing = parallel.get_multiprocessing()
    assert multiprocessing.__name__ == "multiprocessing"


def test_get_multiprocessing_ray():
    parallel.MULTIPROCESSING_BACKEND = "ray"
    multiprocessing = parallel.get_multiprocessing()
    assert multiprocessing.__name__ == "ray.util.multiprocessing"
