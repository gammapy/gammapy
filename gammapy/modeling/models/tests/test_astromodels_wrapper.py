# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
import matplotlib.pyplot as plt
from gammapy.maps import WcsGeom
from gammapy.modeling.models.astromodels_wrapper import (
    AstroPriorModel,
    AstroSpatialModel,
    AstroSpectralModel,
)


def test_spectral():
    model = AstroSpectralModel(function="Super_cutoff_powerlaw")
    energy = 1 * u.TeV
    energy_bounds = [1, 30] * u.TeV

    model.plot(energy_bounds=energy_bounds)
    value = model(energy)

    data = model.to_dict()
    model_new = AstroSpectralModel.from_dict(data=data)
    assert_allclose(model_new(1 * u.TeV), value)
    model_new.plot(energy_bounds=energy_bounds)


def test_spatial():
    model = AstroSpatialModel(
        function="Gaussian_on_sphere", frame="galactic", sigma=0.1
    )
    value = model(0 * u.deg, 0 * u.deg)

    geom = WcsGeom.create(
        skydir=model.position, frame=model.frame, width=(1, 1) * u.deg, binsz=0.02
    )
    plt.figure()
    model.plot(geom=geom)

    data = model.to_dict()
    model_new = AstroSpatialModel.from_dict(data=data)
    assert_allclose(model_new(0 * u.deg, 0 * u.deg), value)
    model_new.plot(geom=geom)


def test_spatial3d():
    model = AstroSpatialModel(
        function="Continuous_injection_diffusion", frame="galactic"
    )
    value = model(0 * u.deg, 0 * u.deg, 1 * u.TeV)

    data = model.to_dict()
    model_new = AstroSpatialModel.from_dict(data=data)
    assert_allclose(model_new(0 * u.deg, 0 * u.deg, 1 * u.TeV), value)


def test_prior():
    model = AstroPriorModel(function="Gaussian")
    value = model(1 * u.Unit(""))

    data = model.to_dict()
    model_new = AstroPriorModel.from_dict(data=data)
    assert_allclose(model_new(1 * u.Unit("")), value)
