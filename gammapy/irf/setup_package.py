# Licensed under a 3-clause BSD style license - see LICENSE.rst


def get_package_data():
    formats = "fits root xml json conf txt csv".split()
    files = ["data/*.{}".format(_) for _ in formats]
    return {"gammapy.irf.tests": files}
