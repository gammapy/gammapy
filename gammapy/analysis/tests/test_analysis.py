# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from pathlib import Path
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from pydantic.error_wrappers import ValidationError
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.datasets import MapDataset, SpectrumDatasetOnOff
from gammapy.maps import WcsGeom, WcsNDMap
from gammapy.modeling.models import DatasetModels
from gammapy.utils.testing import requires_data

CONFIG_PATH = Path(__file__).resolve().parent / ".." / "config"
MODEL_FILE = CONFIG_PATH / "model.yaml"
MODEL_FILE_1D = CONFIG_PATH / "model-1d.yaml"


def get_example_config(which):
    """Example config: which can be 1d or 3d."""
    return AnalysisConfig.read(CONFIG_PATH / f"example-{which}.yaml")


def test_init():
    cfg = {"general": {"outdir": "test"}}
    analysis = Analysis(cfg)
    assert analysis.config.general.outdir == "test"
    with pytest.raises(TypeError):
        Analysis("spam")


def test_update_config():
    analysis = Analysis(AnalysisConfig())
    data = {"general": {"outdir": "test"}}
    config = AnalysisConfig(**data)
    analysis.update_config(config)
    assert analysis.config.general.outdir == "test"

    analysis = Analysis(AnalysisConfig())
    data = """
    general:
        outdir: test
    """
    analysis.update_config(data)
    assert analysis.config.general.outdir == "test"

    analysis = Analysis(AnalysisConfig())
    with pytest.raises(TypeError):
        analysis.update_config(0)


def test_get_observations_no_datastore():
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.config.observations.datastore = "other"
    with pytest.raises(FileNotFoundError):
        analysis.get_observations()


@requires_data()
def test_get_observations_all():
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.config.observations.datastore = "$GAMMAPY_DATA/cta-1dc/index/gps/"
    analysis.get_observations()
    assert len(analysis.observations) == 4


@requires_data()
def test_get_observations_obs_ids():
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.config.observations.datastore = "$GAMMAPY_DATA/cta-1dc/index/gps/"
    analysis.config.observations.obs_ids = ["110380"]
    analysis.get_observations()
    assert len(analysis.observations) == 1


@requires_data()
def test_get_observations_obs_cone():
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.config.observations.datastore = "$GAMMAPY_DATA/hess-dl3-dr1"
    analysis.config.observations.obs_cone = {
        "frame": "icrs",
        "lon": "83d",
        "lat": "22d",
        "radius": "5d",
    }
    analysis.get_observations()
    assert len(analysis.observations) == 4


@requires_data()
def test_get_observations_obs_file(tmp_path):
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.get_observations()
    filename = tmp_path / "obs_ids.txt"
    filename.write_text("20136\n47829\n")
    analysis.config.observations.obs_file = filename
    analysis.get_observations()
    assert len(analysis.observations) == 2


@requires_data()
def test_get_observations_obs_time(tmp_path):
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.config.observations.obs_time = {
        "start": "2004-03-26",
        "stop": "2004-05-26",
    }
    analysis.get_observations()
    assert len(analysis.observations) == 40
    analysis.config.observations.obs_ids = [0]
    with pytest.raises(KeyError):
        analysis.get_observations()


@requires_data()
def test_get_observations_missing_irf():
    config = AnalysisConfig()
    analysis = Analysis(config)
    analysis.config.observations.datastore = "$GAMMAPY_DATA/joint-crab/dl3/magic/"
    analysis.config.observations.obs_ids = ["05029748"]
    analysis.config.observations.required_irf = ["aeff", "edisp"]
    analysis.get_observations()
    assert len(analysis.observations) == 1


@requires_data()
def test_set_models():
    config = get_example_config("3d")
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    models_str = Path(MODEL_FILE).read_text()
    analysis.set_models(models=models_str)
    assert isinstance(analysis.models, DatasetModels)
    assert len(analysis.models) == 2
    assert analysis.models.names == ["source", "stacked-bkg"]
    with pytest.raises(TypeError):
        analysis.set_models(0)

    new_source = analysis.models["source"].copy(name="source2")
    analysis.set_models(models=[new_source], extend=False)
    assert len(analysis.models) == 2
    assert analysis.models.names == ["source2", "stacked-bkg"]


@requires_data()
def test_analysis_1d():
    cfg = """
    observations:
        datastore: $GAMMAPY_DATA/hess-dl3-dr1
        obs_ids: [23523, 23526]
        obs_time: {
            start: [J2004.92654346, J2004.92658453, J2004.92663655],
            stop: [J2004.92658453, J2004.92663655, J2004.92670773]
        }
    datasets:
        type: 1d
        background:
            method: reflected
        geom:
            axes:
                energy_true: {min: 0.01 TeV, max: 300 TeV, nbins: 109}
        on_region: {frame: icrs, lon: 83.633 deg, lat: 22.014 deg, radius: 0.11 deg}
        safe_mask:
            methods: [aeff-default, edisp-bias]
            parameters: {bias_percent: 10.0}
        containment_correction: false
    flux_points:
        energy: {min: 1 TeV, max: 50 TeV, nbins: 4}
    light_curve:
        energy_edges: {min: 1 TeV, max: 50 TeV, nbins: 1}
        time_intervals: {
            start: [J2004.92654346, J2004.92658453, J2004.92663655],
            stop: [J2004.92658453, J2004.92663655, J2004.92670773]
        }
    """
    config = get_example_config("1d")
    analysis = Analysis(config)
    analysis.update_config(cfg)
    analysis.get_observations()
    analysis.get_datasets()
    analysis.read_models(MODEL_FILE_1D)
    analysis.run_fit()
    analysis.get_flux_points()
    analysis.get_light_curve()

    assert len(analysis.datasets) == 3
    table = analysis.flux_points.data.to_table(sed_type="dnde")

    assert len(table) == 4
    dnde = table["dnde"].quantity
    assert dnde.unit == "cm-2 s-1 TeV-1"

    assert_allclose(dnde[0].value, 8.116854e-12, rtol=1e-2)
    assert_allclose(dnde[2].value, 3.444475e-14, rtol=1e-2)

    axis = analysis.light_curve.geom.axes["time"]
    assert axis.nbin == 3
    assert_allclose(axis.time_min.mjd, [53343.92, 53343.935, 53343.954])

    flux = analysis.light_curve.flux.data[:, :, 0, 0]
    assert_allclose(flux, [[1.688954e-11], [2.347870e-11], [1.604152e-11]], rtol=1e-4)


@requires_data()
def test_geom_analysis_1d():
    cfg = """
    observations:
        datastore: $GAMMAPY_DATA/hess-dl3-dr1
        obs_ids: [23523]
    datasets:
        type: 1d
        background:
            method: reflected
        on_region: {frame: icrs, lon: 83.633 deg, lat: 22.014 deg, radius: 0.11 deg}
        geom:
            axes:
                energy: {min: 0.1 TeV, max: 30 TeV, nbins: 20}
                energy_true: {min: 0.03 TeV, max: 100 TeV, nbins: 50}
        containment_correction: false
    flux_points:
        energy: {min: 1 TeV, max: 50 TeV, nbins: 4}
    """
    config = get_example_config("1d")
    analysis = Analysis(config)
    analysis.update_config(cfg)
    analysis.get_observations()
    analysis.get_datasets()

    assert len(analysis.datasets) == 1

    axis = analysis.datasets[0].exposure.geom.axes["energy_true"]
    assert axis.nbin == 50
    assert_allclose(axis.edges[0].to_value("TeV"), 0.03)
    assert_allclose(axis.edges[-1].to_value("TeV"), 100)


@requires_data()
def test_exclusion_region(tmp_path):
    config = get_example_config("1d")
    analysis = Analysis(config)
    region = CircleSkyRegion(center=SkyCoord("85d 23d"), radius=1 * u.deg)
    geom = WcsGeom.create(npix=(150, 150), binsz=0.05, skydir=SkyCoord("83d 22d"))
    exclusion_mask = ~geom.region_mask([region])

    filename = tmp_path / "exclusion.fits"
    exclusion_mask.write(filename)
    config.datasets.background.method = "reflected"
    config.datasets.background.exclusion = filename
    analysis.get_observations()
    analysis.get_datasets()
    assert len(analysis.datasets) == 2

    config = get_example_config("3d")
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    geom = analysis.datasets[0]._geom
    exclusion_mask = ~geom.region_mask([region])
    filename = tmp_path / "exclusion3d.fits"
    exclusion_mask.write(filename)
    config.datasets.background.exclusion = filename
    analysis.get_datasets()
    assert len(analysis.datasets) == 1


@requires_data()
def test_analysis_1d_stacked_no_fit_range():
    cfg = """
    observations:
        datastore: $GAMMAPY_DATA/hess-dl3-dr1
        obs_cone: {frame: icrs, lon: 83.633 deg, lat: 22.014 deg, radius: 5 deg}
        obs_ids: [23592, 23559]

    datasets:
        type: 1d
        stack: false
        geom:
            axes:
                energy: {min: 0.01 TeV, max: 100 TeV, nbins: 73}
                energy_true: {min: 0.03 TeV, max: 100 TeV, nbins: 50}
        on_region: {frame: icrs, lon: 83.633 deg, lat: 22.014 deg, radius: 0.1 deg}
        containment_correction: true
        background:
            method: reflected
    """
    config = AnalysisConfig.from_yaml(cfg)
    analysis = Analysis(config)
    analysis.update_config(cfg)
    analysis.config.datasets.stack = True
    analysis.get_observations()
    analysis.get_datasets()
    analysis.read_models(MODEL_FILE_1D)
    analysis.run_fit()
    with pytest.raises(ValueError):
        analysis.get_excess_map()

    assert len(analysis.datasets) == 1
    assert_allclose(analysis.datasets["stacked"].counts.data.sum(), 184)
    pars = analysis.models.parameters
    assert_allclose(analysis.datasets[0].mask_fit.data, True)

    assert_allclose(pars["index"].value, 2.76913, rtol=1e-2)
    assert_allclose(pars["amplitude"].value, 5.479729e-11, rtol=1e-2)


@requires_data()
def test_analysis_ring_background():
    config = get_example_config("3d")
    config.datasets.background.method = "ring"
    config.datasets.background.parameters = {"r_in": "0.7 deg", "width": "0.7 deg"}
    config.datasets.geom.axes.energy.nbins = 1
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    analysis.get_excess_map()
    assert isinstance(analysis.datasets[0], MapDataset)
    assert_allclose(
        analysis.datasets[0].npred_background().data[0, 10, 10], 0.091799, rtol=1e-2
    )
    assert isinstance(analysis.excess_map["sqrt_ts"], WcsNDMap)
    assert_allclose(analysis.excess_map.npred_excess.data[0, 62, 62], 134.12389)


@requires_data()
def test_analysis_ring_3d():
    config = get_example_config("3d")
    config.datasets.background.method = "ring"
    config.datasets.background.parameters = {"r_in": "0.7 deg", "width": "0.7 deg"}
    analysis = Analysis(config)
    analysis.get_observations()
    with pytest.raises(ValueError):
        analysis.get_datasets()


@requires_data()
def test_analysis_no_bkg_1d(caplog):
    config = get_example_config("1d")
    analysis = Analysis(config)
    with caplog.at_level(logging.WARNING):
        analysis.get_observations()
        analysis.get_datasets()
        assert not isinstance(analysis.datasets[0], SpectrumDatasetOnOff)
        assert "No background maker set. Check configuration." in [
            _.message for _ in caplog.records
        ]


@requires_data()
def test_analysis_no_bkg_3d(caplog):
    config = get_example_config("3d")
    config.datasets.background.method = None
    analysis = Analysis(config)
    with caplog.at_level(logging.WARNING):
        analysis.get_observations()
        analysis.get_datasets()
        assert isinstance(analysis.datasets[0], MapDataset)
        assert "No background maker set. Check configuration." in [
            _.message for _ in caplog.records
        ]


@requires_data()
def test_analysis_3d():
    config = get_example_config("3d")
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    analysis.read_models(MODEL_FILE)
    analysis.datasets["stacked"].background_model.spectral_model.tilt.frozen = False
    analysis.run_fit()
    analysis.get_flux_points()

    assert len(analysis.datasets) == 1
    assert len(analysis.models.parameters) == 8
    res = analysis.models.parameters
    assert res["amplitude"].unit == "cm-2 s-1 TeV-1"

    table = analysis.flux_points.data.to_table(sed_type="dnde")
    assert len(table) == 2
    dnde = table["dnde"].quantity

    assert_allclose(dnde[0].value, 1.2722e-11, rtol=1e-2)
    assert_allclose(dnde[-1].value, 4.054128e-13, rtol=1e-2)
    assert_allclose(res["index"].value, 2.772814, rtol=1e-2)
    assert_allclose(res["tilt"].value, -0.133436, rtol=1e-2)


@requires_data()
def test_analysis_3d_joint_datasets():
    config = get_example_config("3d")
    config.datasets.stack = False
    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    assert len(analysis.datasets) == 2

    assert_allclose(
        analysis.datasets[0].background_model.spectral_model.norm.value,
        1.031743694988066,
        rtol=1e-6,
    )
    assert_allclose(
        analysis.datasets[0].background_model.spectral_model.tilt.value,
        0.0,
        rtol=1e-6,
    )
    assert_allclose(
        analysis.datasets[1].background_model.spectral_model.norm.value,
        0.9776349021876344,
        rtol=1e-6,
    )


@requires_data()
def test_usage_errors():
    config = get_example_config("1d")
    analysis = Analysis(config)
    with pytest.raises(RuntimeError):
        analysis.get_datasets()
    with pytest.raises(RuntimeError):
        analysis.read_datasets()
    with pytest.raises(RuntimeError):
        analysis.write_datasets()
    with pytest.raises(TypeError):
        analysis.read_models()
    with pytest.raises(RuntimeError):
        analysis.write_models()
    with pytest.raises(RuntimeError):
        analysis.run_fit()
    with pytest.raises(RuntimeError):
        analysis.get_flux_points()
    with pytest.raises(ValidationError):
        analysis.config.datasets.type = "None"


@requires_data()
def test_datasets_io(tmpdir):
    config = get_example_config("3d")

    analysis = Analysis(config)
    analysis.get_observations()
    analysis.get_datasets()
    models_str = Path(MODEL_FILE).read_text()
    analysis.models = models_str

    config.general.datasets_file = tmpdir / "datasets.yaml"
    config.general.models_file = tmpdir / "models.yaml"
    analysis.write_datasets()
    analysis = Analysis(config)
    analysis.read_datasets()
    assert len(analysis.datasets.models) == 2
    assert analysis.models.names == ["source", "stacked-bkg"]

    analysis.models[0].parameters["index"].value = 3
    analysis.write_models()
    analysis = Analysis(config)
    analysis.read_datasets()
    assert len(analysis.datasets.models) == 2
    assert analysis.models.names == ["source", "stacked-bkg"]
    assert analysis.models[0].parameters["index"].value == 3
