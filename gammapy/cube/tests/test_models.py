# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data
from ...maps import MapAxis, WcsGeom, Map
from ...irf.energy_dispersion import EnergyDispersion
from ...cube.psf_kernel import PSFKernel
from ...cube.models import SkyDiffuseCube
from ...image.models import SkyGaussian
from ...spectrum.models import PowerLaw
from ..fit import MapEvaluator
from ..models import SkyModel, SkyModels, CompoundSkyModel


@pytest.fixture(scope="session")
def sky_model():
    spatial_model = SkyGaussian(lon_0="3 deg", lat_0="4 deg", sigma="3 deg")
    spectral_model = PowerLaw(
        index=2, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    return SkyModel(spatial_model, spectral_model)


@pytest.fixture(scope="session")
def diffuse_model():
    axis = MapAxis.from_nodes([0.1, 100], name="energy", unit="TeV", interp="log")
    m = Map.create(npix=(4, 3), binsz=2, axes=[axis], unit="cm-2 s-1 MeV-1 sr-1")
    m.data += 42
    return SkyDiffuseCube(m)


@pytest.fixture(scope="session")
def geom():
    axis = MapAxis.from_edges(np.logspace(-1, 1, 3), unit=u.TeV, name="energy")
    return WcsGeom.create(skydir=(0, 0), npix=(5, 4), coordsys="GAL", axes=[axis])


@pytest.fixture(scope="session")
def exposure(geom):
    m = Map.from_geom(geom)
    m.quantity = np.ones((2, 4, 5)) * u.Quantity("100 m2 s")
    m.data[1] *= 10
    return m


@pytest.fixture(scope="session")
def background(geom):
    m = Map.from_geom(geom)
    m.quantity = np.ones((2, 4, 5)) * 1e-7
    return m


@pytest.fixture(scope="session")
def edisp(geom):
    e_true = geom.get_axis_by_name("energy").edges
    return EnergyDispersion.from_diagonal_response(e_true=e_true)


@pytest.fixture(scope="session")
def psf(geom):
    sigma = 0.5 * u.deg
    return PSFKernel.from_gauss(geom, sigma)


@pytest.fixture(scope="session")
def evaluator(sky_model, exposure, background, psf, edisp):
    return MapEvaluator(sky_model, exposure, background, psf=psf, edisp=edisp)


@pytest.fixture(scope="session")
def diffuse_evaluator(diffuse_model, exposure, background, psf, edisp):
    return MapEvaluator(diffuse_model, exposure, background, psf=psf, edisp=edisp)


class TestSkyModels:
    def setup(self):
        self.sky_models = SkyModels([sky_model(), sky_model()])

    def test_to_compound_model(self):
        sky_models = self.sky_models
        model = sky_models.to_compound_model()
        assert isinstance(model, CompoundSkyModel)
        pars = model.parameters.parameters
        assert len(pars) == 12
        assert pars[0].name == "lon_0"
        assert pars[-1].name == "reference"

    def test_parameters(self):
        sky_models = self.sky_models
        parnames = ["lon_0", "lat_0", "sigma", "index", "amplitude", "reference"] * 2
        assert sky_models.parameters.names == parnames

        # Check that model parameters are references to the parts
        p1 = sky_models.parameters["lon_0"]
        p2 = sky_models.skymodels[0].parameters["lon_0"]
        assert p1 is p2

        # Check that parameter assignment works
        assert sky_models.parameters.parameters[-1].value == 1
        sky_models.parameters = sky_models.parameters.copy()
        assert sky_models.parameters.parameters[-1].value == 1

    def test_evaluate(self):
        sky_models = self.sky_models
        lon = 3 * u.deg * np.ones(shape=(3, 4))
        lat = 4 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV

        q = sky_models.evaluate(lon, lat, energy[:, np.newaxis, np.newaxis])

        assert q.unit == "cm-2 s-1 TeV-1 deg-2"
        assert q.shape == (5, 3, 4)
        assert_allclose(q.value, 3.536776513153229e-13)


class TestSkyModel:
    @staticmethod
    def test_repr(sky_model):
        assert "SkyModel" in repr(sky_model)

    @staticmethod
    def test_str(sky_model):
        assert "SkyModel" in str(sky_model)

    @staticmethod
    def test_parameters(sky_model):
        # Check that model parameters are references to the spatial and spectral parts
        p1 = sky_model.parameters["lon_0"]
        p2 = sky_model.spatial_model.parameters["lon_0"]
        assert p1 is p2

        p1 = sky_model.parameters["amplitude"]
        p2 = sky_model.spectral_model.parameters["amplitude"]
        assert p1 is p2

    @staticmethod
    def test_evaluate_scalar(sky_model):
        lon = 3 * u.deg
        lat = 4 * u.deg
        energy = 1 * u.TeV

        q = sky_model.evaluate(lon, lat, energy)

        assert q.unit == "cm-2 s-1 TeV-1 deg-2"
        assert np.isscalar(q.value)
        assert_allclose(q.value, 1.76838826e-13)

    @staticmethod
    def test_evaluate_array(sky_model):
        lon = 3 * u.deg * np.ones(shape=(3, 4))
        lat = 4 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV

        q = sky_model.evaluate(lon, lat, energy[:, np.newaxis, np.newaxis])

        assert q.shape == (5, 3, 4)
        assert_allclose(q.value, 1.76838826e-13)


class TestCompoundSkyModel:
    @staticmethod
    @pytest.fixture()
    def compound_model(sky_model):
        return sky_model + sky_model

    @staticmethod
    def test_parameters(compound_model):
        parnames = ["lon_0", "lat_0", "sigma", "index", "amplitude", "reference"] * 2
        assert compound_model.parameters.names == parnames

        # Check that model parameters are references to the parts
        assert (
            compound_model.parameters["lon_0"]
            is compound_model.model1.parameters["lon_0"]
        )

        # Check that parameter assignment works
        assert compound_model.parameters.parameters[-1].value == 1
        compound_model.parameters = compound_model.parameters.copy()
        assert compound_model.parameters.parameters[-1].value == 1

    @staticmethod
    def test_evaluate(compound_model):
        lon = 3 * u.deg * np.ones(shape=(3, 4))
        lat = 4 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV

        q = compound_model.evaluate(lon, lat, energy[:, np.newaxis, np.newaxis])

        assert q.unit == "cm-2 s-1 TeV-1 deg-2"
        assert q.shape == (5, 3, 4)
        assert_allclose(q.value, 3.536776513153229e-13)


@requires_dependency("scipy")
class TestSkyDiffuseCube:
    @staticmethod
    def test_evaluate_scalar(diffuse_model):
        # Check pixel inside map
        val = diffuse_model.evaluate(0 * u.deg, 0 * u.deg, 10 * u.TeV)
        assert val.unit == "cm-2 s-1 MeV-1 sr-1"
        assert val.shape == (1,)
        assert_allclose(val.value, 42)

        # Check pixel outside map (spatially)
        val = diffuse_model.evaluate(100 * u.deg, 0 * u.deg, 10 * u.TeV)
        assert_allclose(val.value, 0)

        # Check pixel outside energy range
        val = diffuse_model.evaluate(0 * u.deg, 0 * u.deg, 200 * u.TeV)
        assert_allclose(val.value, 0)

    @staticmethod
    def test_evaluate_array(diffuse_model):
        lon = 1 * u.deg * np.ones(shape=(3, 4))
        lat = 2 * u.deg * np.ones(shape=(3, 4))
        energy = [1, 1, 1, 1, 1] * u.TeV

        q = diffuse_model.evaluate(lon, lat, energy[:, np.newaxis, np.newaxis])

        assert q.shape == (5, 3, 4)
        assert_allclose(q.value.mean(), 42)

    @staticmethod
    @requires_data("gammapy-extra")
    def test_read():
        model = SkyDiffuseCube.read(
            "$GAMMAPY_EXTRA/test_datasets/unbundled/fermi/gll_iem_v02_cutout.fits"
        )
        assert model.map.unit == "cm-2 s-1 MeV-1 sr-1"

        # Check pixel inside map
        val = model.evaluate(0 * u.deg, 0 * u.deg, 100 * u.GeV)
        assert val.unit == "cm-2 s-1 MeV-1 sr-1"
        assert val.shape == (1,)
        assert_allclose(val.value, 1.396424e-12, rtol=1e-5)


@requires_dependency("scipy")
class TestSkyDiffuseCubeMapEvaluator:
    @staticmethod
    def test_compute_dnde(diffuse_evaluator):
        out = diffuse_evaluator.compute_dnde()
        assert out.shape == (2, 4, 5)
        out = out.to("cm-2 s-1 MeV-1 sr-1")
        assert_allclose(out.value.sum(), 1680, rtol=1e-5)
        assert_allclose(out.value[0, 0, 0], 42, rtol=1e-5)

    @staticmethod
    def test_compute_flux(diffuse_evaluator):
        out = diffuse_evaluator.compute_flux()
        assert out.shape == (2, 4, 5)
        out = out.to("cm-2 s-1")
        assert_allclose(out.value.sum(), 633263.444803, rtol=1e-5)
        assert_allclose(out.value[0, 0, 0], 2878.196184, rtol=1e-5)

    @staticmethod
    def test_apply_psf(diffuse_evaluator):
        flux = diffuse_evaluator.compute_flux()
        npred = diffuse_evaluator.apply_exposure(flux)
        out = diffuse_evaluator.apply_psf(npred)
        assert out.data.shape == (2, 4, 5)
        assert_allclose(out.data.sum(), 4.004864e+12, rtol=1e-5)
        assert_allclose(out.data[0, 0, 0], 1.380614e+09, rtol=1e-5)

    @staticmethod
    def test_apply_edisp(diffuse_evaluator):
        flux = diffuse_evaluator.compute_flux()
        out = diffuse_evaluator.apply_edisp(flux.value)
        assert out.shape == (2, 4, 5)
        assert_allclose(out.sum(), 633263.444803, rtol=1e-5)
        assert_allclose(out[0, 0, 0], 2878.196184, rtol=1e-5)

    @staticmethod
    def test_compute_npred(diffuse_evaluator):
        out = diffuse_evaluator.compute_npred()
        assert out.shape == (2, 4, 5)
        assert_allclose(out.sum(), 4.004864e+12, rtol=1e-5)
        assert_allclose(out[0, 0, 0], 1.380614e+09, rtol=1e-5)


@requires_dependency("scipy")
class TestSkyModelMapEvaluator:
    @staticmethod
    def test_energy_center(evaluator):
        val = evaluator.energy_center
        assert val.shape == (2, 1, 1)
        assert val.unit == "TeV"

    @staticmethod
    def test_energy_edges(evaluator):
        val = evaluator.energy_edges
        assert val.shape == (3, 1, 1)
        assert val.unit == "TeV"

    @staticmethod
    def test_energy_bin_width(evaluator):
        val = evaluator.energy_bin_width
        assert val.shape == (2, 1, 1)
        assert val.unit == "TeV"

    @staticmethod
    def test_lon_lat(evaluator):
        val = evaluator.lon
        assert val.shape == (4, 5)
        assert val.unit == "deg"

        val = evaluator.lat
        assert val.shape == (4, 5)
        assert val.unit == "deg"

    @staticmethod
    def test_solid_angle(evaluator):
        val = evaluator.solid_angle
        assert val.shape == (2, 4, 5)
        assert val.unit == "sr"

    @staticmethod
    def test_bin_volume(evaluator):
        val = evaluator.bin_volume
        assert val.shape == (2, 4, 5)
        assert val.unit == "TeV sr"

    @staticmethod
    def test_compute_dnde(evaluator):
        out = evaluator.compute_dnde()
        assert out.shape == (2, 4, 5)
        assert out.unit == "cm-2 s-1 TeV-1 deg-2"
        assert_allclose(out.value.sum(), 2.984368e-12, rtol=1e-5)
        assert_allclose(out.value[0, 0, 0], 1.336901e-13, rtol=1e-5)

    @staticmethod
    def test_compute_flux(evaluator):
        out = evaluator.compute_flux()
        assert out.shape == (2, 4, 5)
        assert out.unit == "cm-2 s-1"
        assert_allclose(out.value.sum(), 7.312833e-13, rtol=1e-5)
        assert_allclose(out.value[0, 0, 0], 3.007569e-14, rtol=1e-5)

    @staticmethod
    def test_apply_psf(evaluator):
        flux = evaluator.compute_flux()
        npred = evaluator.apply_exposure(flux)
        out = evaluator.apply_psf(npred)
        assert out.data.shape == (2, 4, 5)
        assert_allclose(out.data.sum(), 9.144771e-07, rtol=1e-5)
        assert_allclose(out.data[0, 0, 0], 1.563604e-08, rtol=1e-5)

    @staticmethod
    def test_apply_edisp(evaluator):
        flux = evaluator.compute_flux()
        out = evaluator.apply_edisp(flux.value)
        assert out.shape == (2, 4, 5)
        assert_allclose(out.sum(), 7.312833e-13, rtol=1e-5)
        assert_allclose(out[0, 0, 0], 3.007569e-14, rtol=1e-5)

    @staticmethod
    def test_compute_npred(evaluator):
        out = evaluator.compute_npred()
        assert out.shape == (2, 4, 5)
        assert_allclose(out.sum(), 4.914477e-06, rtol=1e-5)
        assert_allclose(out[0, 0, 0], 1.15636e-07, rtol=1e-5)
