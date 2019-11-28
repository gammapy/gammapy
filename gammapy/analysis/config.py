# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import List
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity
import yaml
from pydantic import BaseModel, FilePath
from pydantic.utils import deep_update
from gammapy.utils.scripts import read_yaml, make_path

__all__ = ["AnalysisConfig"]

CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"
ANALYSIS_TEMPLATES = {"1d": "template-1d.yaml", "3d": "template-3d.yaml"}

log = logging.getLogger(__name__)


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


class ReductionTypeEnum(str, Enum):
    spectrum = "1d"
    cube = "3d"


class FrameEnum(str, Enum):
    icrs = "icrs"
    galactic = "galactic"


class BackgroundMethodEnum(str, Enum):
    reflected = "reflected"


class GammapyBaseConfig(BaseModel):
    class Config:
        validate_all = True
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            Angle: lambda v: f"{v.value} {v.unit}",
            Quantity: lambda v: f"{v.value} {v.unit}",
            Time: lambda v: f"{v.value}",
        }


class SkyCoordConfig(GammapyBaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None


class EnergyAxisConfig(GammapyBaseConfig):
    min: EnergyType = "0.1 TeV"
    max: EnergyType = "10 TeV"
    nbins: int = 30


class SpatialCircleRangeConfig(GammapyBaseConfig):
    frame: FrameEnum = None
    lon: AngleType = None
    lat: AngleType = None
    radius: AngleType = None


class EnergyRangeConfig(GammapyBaseConfig):
    min: EnergyType = "0.1 TeV"
    max: EnergyType = "10 TeV"


class TimeRangeConfig(GammapyBaseConfig):
    start: TimeType = None
    stop: TimeType = None


class FluxPointsConfig(GammapyBaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig()


class FitConfig(GammapyBaseConfig):
    fit_range: EnergyRangeConfig = EnergyRangeConfig()


class BackgroundConfig(GammapyBaseConfig):
    method: BackgroundMethodEnum = BackgroundMethodEnum.reflected
    exclusion: FilePath = None


class AxesConfig(GammapyBaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig()
    energy_true: EnergyAxisConfig = EnergyAxisConfig()


class SelectionConfig(GammapyBaseConfig):
    offset_max: AngleType = "2.5 deg"


class FovConfig(GammapyBaseConfig):
    width: AngleType = "5 deg"
    height: AngleType = "5 deg"


class WcsConfig(GammapyBaseConfig):
    skydir: SkyCoordConfig = SkyCoordConfig()
    binsize: AngleType = "0.1 deg"
    fov: FovConfig = FovConfig()
    binsize_irf: AngleType = "0.1 deg"
    margin_irf: AngleType = "0.1 deg"


class GeomConfig(GammapyBaseConfig):
    wcs: WcsConfig = WcsConfig()
    selection: SelectionConfig = SelectionConfig()
    axes: AxesConfig = AxesConfig()


class DatasetsConfig(GammapyBaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    stack: bool = True
    geom: GeomConfig = GeomConfig()
    background: BackgroundConfig = BackgroundConfig()
    on_region: SpatialCircleRangeConfig = SpatialCircleRangeConfig()
    containment_correction: bool = True
    psf_kernel_radius: AngleType = "0.6 deg"


class ObservationsConfig(GammapyBaseConfig):
    datastore: Path = Path("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids: List[int] = []
    obs_file: FilePath = None
    obs_cone: SpatialCircleRangeConfig = SpatialCircleRangeConfig()
    obs_time: TimeRangeConfig = TimeRangeConfig()


class LogConfig(GammapyBaseConfig):
    level: str = "info"
    filename: Path = None
    filemode: str = None
    format: str = None
    datefmt: str = None


class GeneralConfig(GammapyBaseConfig):
    log: LogConfig = LogConfig()
    outdir: str = "."


class AnalysisConfig(GammapyBaseConfig):
    """Config class handling the high-level interface settings."""

    general: GeneralConfig = GeneralConfig()
    observations: ObservationsConfig = ObservationsConfig()
    datasets: DatasetsConfig = DatasetsConfig()
    fit: FitConfig = FitConfig()
    flux_points: FluxPointsConfig = FluxPointsConfig()

    @classmethod
    def from_template(cls, template):
        """Create AnalysisConfig from existing templates.

        Parameters
        ----------
        template : {"1d", "3d"}
            Built-in templates.

        Returns
        -------
        config : `AnalysisConfig`
            AnalysisConfig class
        """
        filename = CONFIG_PATH / ANALYSIS_TEMPLATES[template]
        return cls.from_yaml(filename=filename)

    def __str__(self):
        """Display settings in pretty YAML format."""
        info = self.__class__.__name__ + "\n\n\t"

        data = self.to_yaml()
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def from_yaml(cls, settings="", filename=None):
        config = yaml.safe_load(settings)
        if filename:
            config = read_yaml(filename)
        return AnalysisConfig(**config)

    def to_yaml(self, filename=None, overwrite=False):
        if filename:
            fname = Path(filename).name
            fpath = Path(self.general.outdir) / fname
            if fpath.exists() and not overwrite:
                raise IOError(f"File {filename} already exists.")
            fpath.write_text(
                yaml.dump(yaml.safe_load(self.json()), sort_keys=False, indent=4)
            )
            log.info(f"Configuration settings saved into {fpath}")
        else:
            return yaml.dump(yaml.safe_load(self.json()), sort_keys=False, indent=4)

    def set_logging(self):
        """Set logging parameters for API."""
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.dict())
        log.info("Setting logging config: {!r}".format(self.general.log.dict()))

    def update(self, config=None, filename=""):
        """Updates config with provided settings.

         Parameters
         ----------
         config : string dict or `AnalysisConfig` object
             Configuration settings provided in dict() syntax.
         filename : string
             Filename in YAML format.
         """
        try:
            if filename:
                filepath = make_path(filename)
                config = dict(read_yaml(filepath))
            if isinstance(config, str) and not filename:
                config = dict(yaml.safe_load(config))
        except Exception:
            raise ValueError("Could not parse the config settings provided.")
        if isinstance(config, dict):
            config = AnalysisConfig(**config)
        if isinstance(config, AnalysisConfig):
            upd_config = deep_update(
                self.dict(exclude_defaults=True), config.dict(exclude_defaults=True)
            )
            return AnalysisConfig(**upd_config)

    def help(self, section=""):
        """Print template configuration settings."""
        doc = self._get_doc_sections()
        for keyword in doc.keys():
            if section == "" or section == keyword:
                print(doc[keyword])

    @staticmethod
    def _get_doc_sections():
        """Returns dict with commented docs from docs file"""
        doc = defaultdict(str)
        with open(DOCS_FILE) as f:
            for line in filter(lambda line: not line.startswith("---"), f):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc
