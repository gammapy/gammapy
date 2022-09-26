# Licensed under a 3-clause BSD style license - see LICENSE.rst
# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
import os
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib
from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
from gammapy.data import GTI
from gammapy.datasets import SpectrumDataset
from gammapy.maps import MapAxis, RegionNDMap
from gammapy.modeling.models import (
    ConstantTemporalModel,
    Models,
    PowerLawSpectralModel,
    SkyModel,
)

# TODO: activate this again and handle deprecations in the code
# enable_deprecations_as_exceptions(warnings_to_ignore_entire_module=["iminuit", "naima"])


def pytest_configure(config):
    """Print some info ..."""
    from gammapy.utils.testing import has_data

    config.option.astropy_header = True

    # Declare for which packages version numbers should be displayed
    # when running the tests
    PYTEST_HEADER_MODULES["cython"] = "cython"
    PYTEST_HEADER_MODULES["iminuit"] = "iminuit"
    PYTEST_HEADER_MODULES["matplotlib"] = "matplotlib"
    PYTEST_HEADER_MODULES["astropy"] = "astropy"
    PYTEST_HEADER_MODULES["regions"] = "regions"
    PYTEST_HEADER_MODULES["healpy"] = "healpy"
    PYTEST_HEADER_MODULES["sherpa"] = "sherpa"
    PYTEST_HEADER_MODULES["gammapy"] = "gammapy"
    PYTEST_HEADER_MODULES["naima"] = "naima"

    print("")
    print("Gammapy test data availability:")

    has_it = "yes" if has_data("gammapy-data") else "no"
    print(f"gammapy-data ... {has_it}")

    print("Gammapy environment variables:")

    var = os.environ.get("GAMMAPY_DATA", "not set")
    print(f"GAMMAPY_DATA = {var}")

    matplotlib.use("agg")
    print('Setting matplotlib backend to "agg" for the tests.')

    from . import __version__

    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = __version__


@pytest.fixture()
def spectrum_dataset():
    # TODO: change the fixture scope to "session". This currently crashes fitting tests
    name = "test"
    energy = np.logspace(-1, 1, 31) * u.TeV
    livetime = 100 * u.s

    pwl = PowerLawSpectralModel(
        index=2.1,
        amplitude="1e5 cm-2 s-1 TeV-1",
        reference="0.1 TeV",
    )

    temp_mod = ConstantTemporalModel()

    model = SkyModel(spectral_model=pwl, temporal_model=temp_mod, name="test-source")
    axis = MapAxis.from_edges(energy, interp="log", name="energy")
    axis_true = MapAxis.from_edges(energy, interp="log", name="energy_true")

    background = RegionNDMap.create(region="icrs;circle(0, 0, 0.1)", axes=[axis])

    models = Models([model])
    exposure = RegionNDMap.create(region="icrs;circle(0, 0, 0.1)", axes=[axis_true])
    exposure.quantity = u.Quantity("1 cm2") * livetime
    bkg_rate = np.ones(30) / u.s
    background.quantity = bkg_rate * livetime

    start = [1, 3, 5] * u.day
    stop = [2, 3.5, 6] * u.day
    t_ref = Time(55555, format="mjd")
    gti = GTI.create(start, stop, reference_time=t_ref)

    dataset = SpectrumDataset(
        models=models,
        exposure=exposure,
        background=background,
        name=name,
        gti=gti,
    )
    dataset.fake(random_state=23)
    return dataset
