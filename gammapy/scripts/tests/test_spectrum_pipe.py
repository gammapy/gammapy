# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion
from ...utils.energy import energy_logspace
from ...utils.testing import requires_dependency, requires_data
from ...spectrum.models import PowerLaw
from ..spectrum_pipe import SpectrumAnalysisIACT
from ...data import DataStore


@pytest.fixture(scope="session")
def observations():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs_ids = [23523, 23526]
    return datastore.get_observations(obs_ids)


@pytest.fixture(scope="session")
def config():
    """Get test config, extend to several scenarios"""
    model = PowerLaw(
        index=2, amplitude=1e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1 * u.TeV
    )

    pos = SkyCoord(83.63, 22.01, unit="deg", frame="icrs")
    radius = Angle(0.11, "deg")
    on_region = CircleSkyRegion(pos, radius)
    fp_binning = energy_logspace(1, 50, 5, "TeV")
    return dict(
        outdir=None,
        background=dict(on_region=on_region),
        extraction=dict(),
        fit=dict(model=model),
        fp_binning=fp_binning,
    )


@requires_data()
@requires_dependency("sherpa")
def test_spectrum_analysis_iact(tmpdir, config, observations):
    config["outdir"] = tmpdir

    analysis = SpectrumAnalysisIACT(observations=observations, config=config)
    analysis.run()
    flux_points = analysis.flux_points

    assert len(flux_points.table) == 4

    dnde = flux_points.table["dnde"].quantity
    dnde.unit == "cm-2 s-1 TeV-1"
    assert_allclose(dnde[0].value, 6.601518e-12, rtol=1e-2)
    assert_allclose(dnde[-1].value, 1.295918e-15, rtol=1e-2)
