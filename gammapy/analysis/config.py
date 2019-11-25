from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
from gammapy.utils.scripts import make_path, read_yaml
from pydantic import BaseModel, FilePath, validator
from pydantic.utils import deep_update
from pathlib import Path
from typing import List
from enum import Enum
import yaml

__all__ = ["Config", "General"]


class GammapyBaseModel(BaseModel):
    class Config:
        validate_assignment = True
        extra = 'forbid'

    @classmethod
    def from_yaml(cls, filename):
        config = read_yaml(filename)
        return Config(**config)

    def to_yaml(self):
        return yaml.dump(self.dict())

    def update_from_dict(self, other):
        data = deep_update(self.dict(exclude_defaults=True), other.dict(exclude_defaults=True))
        return Config(**data)


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
        assert isinstance(Quantity(v).to("erg"), Quantity)
        return Quantity(v)


class TimeType(Time):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Time(v)


class FrameEnum(str, Enum):
    icrs = 'icrs'
    galactic = 'galactic'


class BackgroundMethodEnum(str, Enum):
    reflected = 'reflected'


class Skydir(GammapyBaseModel):
    frame: FrameEnum = FrameEnum.icrs
    lon: AngleType = AngleType("83.633 deg")
    lat: AngleType = AngleType("22.014 deg")


class EnergyAxis(GammapyBaseModel):
    min: EnergyType = EnergyType("0.1 TeV")
    max: EnergyType = EnergyType("10 TeV")
    nbins: int = 30


class SpatialCircleRange(GammapyBaseModel):
    frame: FrameEnum = FrameEnum.icrs
    lon: AngleType = AngleType("83.633 deg")
    lat: AngleType = AngleType("22.014 deg")
    radius: AngleType = AngleType("0.1 deg")


class EnergyRange(GammapyBaseModel):
    min: EnergyType = EnergyType("0.1 TeV")
    max: EnergyType = EnergyType("10 TeV")


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
    offset_max: AngleType = AngleType("2.5 deg")


class Fov(GammapyBaseModel):
    width: AngleType = AngleType("5 deg")
    height: AngleType = AngleType("5 deg")


class Wcs(GammapyBaseModel):
    skydir: Skydir = Skydir()
    binsize: AngleType = AngleType("0.1 deg")
    fov: Fov = Fov()
    binsize_irf: AngleType = AngleType("0.1 deg")
    margin_irf: AngleType = AngleType("0.1 deg")


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
    psf_kernel_radius: AngleType = AngleType("0.6 deg")


class Data(GammapyBaseModel):
    datastore: Path = "$GAMMAPY_DATA/hess-dl3-dr1/"
    obs_ids: List[int] = []
    obs_file: FilePath = None
    obs_cone: SpatialCircleRange = SpatialCircleRange()
    obs_time: TimeRange = TimeRange()

    @validator("datastore")
    def datastore_exists(cls, v):
        return make_path(v)


class Log(GammapyBaseModel):
    level: str = "info"
    filename: Path = None
    filemode: str = None
    format: str = None
    datefmt: str = None


class General(GammapyBaseModel):
    log: Log = Log()
    outdir: str = "."


class Config(GammapyBaseModel):
    """Config class handling the high-level interface settings."""
    general: General = General()
    data: Data = Data()
    datasets: Datasets = Datasets()
    fit: Fit = Fit()
    flux_points: FluxPoints = FluxPoints()
