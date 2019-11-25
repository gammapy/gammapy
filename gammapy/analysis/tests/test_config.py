from pathlib import Path
import pytest
from astropy.units import Quantity
from gammapy.analysis.config import (
    AnalysisConfig,
    AngleType,
    Axes,
    Background,
    BackgroundMethodEnum,
    Data,
    Datasets,
    EnergyAxis,
    EnergyRange,
    EnergyType,
    Fit,
    FluxPoints,
    Fov,
    FrameEnum,
    General,
    Geom,
    Log,
    Selection,
    SkyCoordType,
    SpatialCircleRange,
    TimeRange,
    TimeType,
    Wcs,
)

config_file = Path(__file__).resolve().parent / ".." / "config" / "config.yaml"


def test_config_basics():
    config = AnalysisConfig()
    assert isinstance(config.general, General)
    assert isinstance(config.general.log, Log)
    assert isinstance(config.data, Data)
    assert isinstance(config.datasets, Datasets)
    assert isinstance(config.datasets.geom, Geom)
    assert isinstance(config.datasets.geom.wcs, Wcs)
    assert isinstance(config.datasets.geom.wcs.fov, Fov)
    assert isinstance(config.datasets.geom.selection, Selection)
    assert isinstance(config.datasets.geom.axes, Axes)
    assert isinstance(config.datasets.background, Background)
    assert isinstance(config.fit, Fit)
    assert isinstance(config.flux_points, FluxPoints)
    assert isinstance(config.data.obs_time, TimeRange)
    assert isinstance(config.fit.fit_range, EnergyRange)
    assert isinstance(config.data.obs_cone, SpatialCircleRange)
    assert isinstance(config.flux_points.energy, EnergyAxis)
    config.datasets.geom.wcs.skydir = {
        "frame": "galactic",
        "lon": "83.633 deg",
        "lat": "22.014 deg",
    }
    assert isinstance(config.datasets.geom.wcs.skydir, SkyCoordType)
    assert isinstance(config.datasets.background.method, BackgroundMethodEnum)
    assert isinstance(config.datasets.geom.wcs.skydir.frame, FrameEnum)
    config.data.obs_time.start = "2019-12-01"
    assert isinstance(config.data.obs_time.start, TimeType)
    assert isinstance(config.fit.fit_range.min, EnergyType)
    assert isinstance(config.datasets.geom.wcs.binsize, AngleType)


def test_config_create_from_dict():
    data = {"general": {"log": {"level": "warning"}}}
    config = AnalysisConfig(**data)
    assert config.general.log.level == "warning"


def test_config_create_from_yaml():
    config = AnalysisConfig.from_yaml(config_file)
    assert isinstance(config.general, General)


def test_config_to_yaml():
    config = AnalysisConfig()
    assert "level: info" in config.to_yaml()


def test_config_update_from_dict():
    config1 = AnalysisConfig()
    data = {"fit": {"fit_range": {"min": "1 TeV", "max": "100 TeV"}}}
    config2 = AnalysisConfig(**data)
    config = config1.update_from_dict(config2)
    assert config.fit.fit_range.min == Quantity("1 TeV")
    assert config.general.log.level == "info"
