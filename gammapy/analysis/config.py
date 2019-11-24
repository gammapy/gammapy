from typing import List
import yaml
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.units import Quantity
from pydantic import BaseModel

__all__ = ["Config", "General"]


class GammapyBaseModel(BaseModel):
    class Config:
        validate_assignment = True
        extra = 'forbid'

    def to_yaml(self):
        return yaml.dump(self.dict())

    def update_from_dict(self):
        pass


class FileNameType(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    # TODO
    # filename exists
    @classmethod
    def validate(cls, v):
        pass


class FrameType(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    # TODO
    # frame is a valid frame
    @classmethod
    def validate(cls, v):
        pass


class BackgroundMethodType(str):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    # TODO
    # only reflected method allowed
    @classmethod
    def validate(cls, v):
        pass


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
        assert isinstance(v.to("erg"), Quantity)
        return Quantity(v)


class TimeType(Time):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return Time(v)


class Skydir(GammapyBaseModel):
    frame: FrameType = FrameType("icrs")
    lon: AngleType = AngleType("83.633 deg")
    lat: AngleType = AngleType("22.014 deg")


class EnergyAxis(GammapyBaseModel):
    min: EnergyRange = EnergyType("0.1 TeV")
    max: EnergyRange = EnergyType("10 TeV")
    nbins: int = 30


class SpatialCircleRange(GammapyBaseModel):
    frame: FrameType = FrameType("icrs")
    lon: AngleType = AngleType("83.633 deg")
    lat: AngleType = AngleType("22.014 deg")
    radius: AngleType = AngleType("0.1 deg")


class EnergyRange(GammapyBaseModel):
    min: EnergyType = EnergyType("0.1 TeV")
    max: EnergyType = EnergyType("10 TeV")


class TimeRange(GammapyBaseModel):
    start: TimeType = None
    stop: TimeType = None

    # TODO
    # stop bigger than start
    @classmethod
    def validate(cls, v):
        pass


class FluxPoints(GammapyBaseModel):
    fit_range: EnergyAxis = EnergyAxis()


class Fit(GammapyBaseModel):
    fit_range: EnergyRange = EnergyRange()


class Background(GammapyBaseModel):
    method: BackgroundMethodType = BackgroundMethodType("reflected")
    exclusion: FileNameType = None


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
    onregion: SpatialCircleRange = None
    containment_correction: bool = True
    psf_kernel_radius: AngleType = AngleType("0.6 deg")


class Data(GammapyBaseModel):
    datastore: str = "$GAMMAPY_DATA/hess-dl3-dr1/"
    obs_ids = List[int] = []
    obs_file = str = None
    obs_cone = SpatialCircleRange = None
    obs_time = TimeRange = None

    # TODO
    # mutually exclusive obs_ids filters
    @classmethod
    def validate(cls, v):
        pass


class Log(GammapyBaseModel):
    level: str = "info"
    filename: str = None
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
