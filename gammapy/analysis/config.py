# Licensed under a 3-clause BSD style license - see LICENSE.rst
from enum import Enum
from pathlib import Path
from typing import List
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
import yaml
from pydantic import BaseModel, FilePath
from pydantic.utils import deep_update
from gammapy.utils.scripts import read_yaml

__all__ = ["AnalysisConfig"]


class AngleType(Angle):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Angle(v)


class EnergyType(Quantity):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        v = Quantity(v)
        if v.unit.physical_type != "energy":
            raise ValueError(f"Invalid unit for energy: {v.unit!r}")
        return v


class TimeType(Time):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Time(v)


class FrameEnum(str, Enum):
    icrs = "icrs"
    galactic = "galactic"


class BackgroundMethodEnum(str, Enum):
    reflected = "reflected"


class GammapyBaseModel(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            Angle: lambda v: f"{v.value} {v.unit}",
            Quantity: lambda v: f"{v.value} {v.unit}",
            Time: lambda v: f"{v.value}",
        }

    @classmethod
    def from_yaml(cls, filename):
        config = read_yaml(filename)
        return AnalysisConfig(**config)

    def to_yaml(self):
        return yaml.dump(yaml.safe_load(self.json()))

    def update_from_dict(self, other):
        data = deep_update(
            self.dict(exclude_defaults=True), other.dict(exclude_defaults=True)
        )
        return AnalysisConfig(**data)


class SkyCoordType(GammapyBaseModel):
    frame: FrameEnum = FrameEnum.icrs
    lon: AngleType = None
    lat: AngleType = None


class EnergyAxis(GammapyBaseModel):
    min: EnergyType = "0.1 TeV"
    max: EnergyType = "10 TeV"
    nbins: int = 30


class SpatialCircleRange(GammapyBaseModel):
    frame: FrameEnum = FrameEnum.icrs
    lon: AngleType = None
    lat: AngleType = None
    radius: AngleType = None


class EnergyRange(GammapyBaseModel):
    min: EnergyType = "0.1 TeV"
    max: EnergyType = "10 TeV"


class TimeRange(GammapyBaseModel):
    start: TimeType = None
    stop: TimeType = None


class FluxPoints(GammapyBaseModel):
    energy: EnergyAxis = EnergyAxis()


class Fit(GammapyBaseModel):
    fit_range: EnergyRange = EnergyRange()


class Background(GammapyBaseModel):
    method: BackgroundMethodEnum = BackgroundMethodEnum.reflected
    exclusion: FilePath = None


class Axes(GammapyBaseModel):
    energy: EnergyAxis = EnergyAxis()
    energy_true: EnergyAxis = EnergyAxis()


class Selection(GammapyBaseModel):
    offset_max: AngleType = "2.5 deg"


class Fov(GammapyBaseModel):
    width: AngleType = "5 deg"
    height: AngleType = "5 deg"


class Wcs(GammapyBaseModel):
    skydir: SkyCoordType = SkyCoordType()
    binsize: AngleType = "0.1 deg"
    fov: Fov = Fov()
    binsize_irf: AngleType = "0.1 deg"
    margin_irf: AngleType = "0.1 deg"


class Geom(GammapyBaseModel):
    wcs: Wcs = Wcs()
    selection: Selection = Selection()
    axes: Axes = Axes()


class Datasets(GammapyBaseModel):
    type: str = "1d"
    stack: bool = True
    geom: Geom = Geom()
    background: Background = Background()
    onregion: SpatialCircleRange = SpatialCircleRange()
    containment_correction: bool = True
    psf_kernel_radius: AngleType = "0.6 deg"


class Data(GammapyBaseModel):
    datastore: Path = Path("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids: List[int] = []
    obs_file: FilePath = None
    obs_cone: SpatialCircleRange = SpatialCircleRange()
    obs_time: TimeRange = TimeRange()


class Log(GammapyBaseModel):
    level: str = "info"
    filename: Path = None
    filemode: str = None
    format: str = None
    datefmt: str = None


class General(GammapyBaseModel):
    log: Log = Log()
    outdir: str = "."


class AnalysisConfig(GammapyBaseModel):
    """Config class handling the high-level interface settings."""

    general: General = General()
    data: Data = Data()
    datasets: Datasets = Datasets()
    fit: Fit = Fit()
    flux_points: FluxPoints = FluxPoints()
