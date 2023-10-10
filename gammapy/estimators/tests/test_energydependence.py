import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.angle_utilities import angular_separation
from gammapy.data import Observation, observatory_locations
from gammapy.data.pointing import FixedPointingInfo, PointingMode
from gammapy.datasets import Datasets, MapDataset
from gammapy.estimators.energydependence import (
    EnergyDependenceEstimator,
    weighted_chi2_parameter,
)
from gammapy.irf import load_irf_dict_from_file
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.maps import MapAxis, WcsGeom
from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    SpatialModel,
)


class MyCustomGaussianModel(SpatialModel):
    """My custom Energy Dependent Gaussian model.

    Parameters
    ----------
    lon_0, lat_0 : `~astropy.coordinates.Angle`
        Center position
    sigma_1TeV : `~astropy.coordinates.Angle`
        Width of the Gaussian at 1 TeV
    sigma_10TeV : `~astropy.coordinates.Angle`
        Width of the Gaussian at 10 TeV

    """

    tag = "MyCustomGaussianModel"
    is_energy_dependent = True
    lon_0 = Parameter("lon_0", "5.6 deg")
    lat_0 = Parameter("lat_0", "0.2 deg", min=-90, max=90)

    sigma_1TeV = Parameter("sigma_1TeV", "0.3 deg", min=0)
    sigma_10TeV = Parameter("sigma_10TeV", "0.15 deg", min=0)

    SpatialModel.frame = "galactic"

    @staticmethod
    def evaluate(lon, lat, energy, lon_0, lat_0, sigma_1TeV, sigma_10TeV):
        sep = angular_separation(lon, lat, lon_0, lat_0)

        # Compute sigma for the given energy using linear interpolation in log energy
        sigma_nodes = u.Quantity([sigma_1TeV, sigma_10TeV])
        energy_nodes = [1, 10] * u.TeV
        log_s = np.log(sigma_nodes.to("deg").value)
        log_en = np.log(energy_nodes.to("TeV").value)
        log_e = np.log(energy.to("TeV").value)
        sigma = np.exp(np.interp(log_e, log_en, log_s)) * u.deg

        exponent = -0.5 * (sep / sigma) ** 2
        norm = 1 / (2 * np.pi * sigma**2)

        return norm * np.exp(exponent)

    @property
    def evaluation_radius(self):
        """Evaluation radius (`~astropy.coordinates.Angle`)."""
        return 2 * np.max([self.sigma_1TeV.value, self.sigma_10TeV.value]) * u.deg


def create_dataset():
    source_lat = 0.2
    source_lon = 5.6
    spatial_model = MyCustomGaussianModel()

    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="model1"
    )

    irfs = load_irf_dict_from_file(
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    livetime = 10.0 * u.hr
    location = observatory_locations["cta_south"]
    # TO DO: update to remove 'mode' in 1.3
    pointing = FixedPointingInfo(
        mode=PointingMode.POINTING,
        fixed_icrs=SkyCoord(source_lon, source_lat, unit="deg", frame="galactic").icrs,
    )

    obs = Observation.create(
        pointing=pointing,
        livetime=livetime,
        irfs=irfs,
        location=location,
    )

    # Define map geometry for binned simulation
    energy_reco = MapAxis.from_edges(
        np.logspace(-1.0, 2, 20), unit="TeV", name="energy", interp="log"
    )
    geom = WcsGeom.create(
        skydir=(source_lon, source_lat),
        binsz=0.02,
        width=5 * u.deg,
        frame="galactic",
        axes=[energy_reco],
    )

    # Make the MapDataset
    empty = MapDataset.create(geom, name="dataset-simu")
    maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
    maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=2.0 * u.deg)
    dataset = maker.run(empty, obs)
    dataset = maker_safe_mask.run(dataset, obs)

    # Add the model on the dataset and Poission fluctuate
    dataset.models = model
    dataset.fake(random_state=42)

    return dataset


source_pos = SkyCoord(5.58, 0.2, unit="deg", frame="galactic")
energy_edges = [1, 3, 5, 20] * u.TeV
spectral_model = PowerLawSpectralModel(
    index=2.94, amplitude=9.8e-12 * u.Unit("cm-2 s-1 TeV-1"), reference=1.0 * u.TeV
)
spatial_model = GaussianSpatialModel(
    lon_0=source_pos.l, lat_0=source_pos.b, frame="galactic", sigma=0.2 * u.deg
)

model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model, name="source"
)

model.spatial_model.lon_0.frozen = False
model.spatial_model.lat_0.frozen = False
model.spatial_model.sigma.frozen = False

model.spectral_model.amplitude.frozen = False
model.spectral_model.index.frozen = True

spatial_model.lon_0.min = source_pos.galactic.l.deg - 0.8
spatial_model.lon_0.max = source_pos.galactic.l.deg + 0.8
spatial_model.lat_0.min = source_pos.galactic.b.deg - 0.8
spatial_model.lat_0.max = source_pos.galactic.b.deg + 0.8

dataset = create_dataset()
dataset.models = model

datasets = Datasets([dataset])

estimator = EnergyDependenceEstimator(energy_edges=energy_edges, source="source")
results = estimator.run(datasets)


def test_edep():
    results_edep = results["energy_dependence"]["result"]
    assert_allclose(
        results_edep["lon_0"],
        [5.59758936, 5.59883752, 5.59508081, 5.60370771] * u.deg,
        atol=1e-5,
    )
    assert_allclose(
        results_edep["lat_0"],
        [0.19058239, 0.20215059, 0.18426731, 0.17378267] * u.deg,
        atol=1e-5,
    )
    assert_allclose(
        results_edep["sigma"],
        [0.22264514, 0.25998229, 0.1891126, 0.18796409] * u.deg,
        atol=1e-5,
    )
    assert_allclose(
        results["energy_dependence"]["delta_ts"], 56.38850999822898, atol=1e-5
    )


def test_significance():
    results_src = results["src_above_bkg"]
    assert_allclose(
        results_src["delta_ts"],
        [1546.2458570901, 1168.6506571890714, 435.2570648315741],
        atol=1e-5,
    )
    assert_allclose(
        results_src["significance"],
        [np.inf, 33.88815235691276, 20.44481642577318],
        atol=1e-5,
    )


def test_chi2():
    results_edep = results["energy_dependence"]["result"]
    chi2_sigma = weighted_chi2_parameter(results_edep, parameter="sigma")
    assert_allclose(chi2_sigma["chi2 sigma"], [52.510946417517644], atol=1e-5)
    assert_allclose(chi2_sigma["significance"], [6.938700011859214], atol=1e-5)
