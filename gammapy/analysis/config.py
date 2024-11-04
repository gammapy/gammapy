# Licensed under a 3-clause BSD style license - see LICENSE.rst
import html
import json
import logging
from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, ConfigDict
from gammapy.makers import MapDatasetMaker
from gammapy.utils.scripts import read_yaml, to_yaml, write_yaml
from gammapy.utils.types import AngleType, EnergyType, PathType, TimeType

__all__ = ["AnalysisConfig"]

CONFIG_PATH = Path(__file__).resolve().parent / "config"
DOCS_FILE = CONFIG_PATH / "docs.yaml"

log = logging.getLogger(__name__)


def deep_update(d, u):
    """Recursively update a nested dictionary.

    Taken from: https://stackoverflow.com/a/3233356/19802442
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class ReductionTypeEnum(str, Enum):
    spectrum = "1d"
    cube = "3d"


class FrameEnum(str, Enum):
    icrs = "icrs"
    galactic = "galactic"


class RequiredHDUEnum(str, Enum):
    events = "events"
    gti = "gti"
    aeff = "aeff"
    bkg = "bkg"
    edisp = "edisp"
    psf = "psf"
    rad_max = "rad_max"


class BackgroundMethodEnum(str, Enum):
    reflected = "reflected"
    fov = "fov_background"
    ring = "ring"


class SafeMaskMethodsEnum(str, Enum):
    aeff_default = "aeff-default"
    aeff_max = "aeff-max"
    edisp_bias = "edisp-bias"
    offset_max = "offset-max"
    bkg_peak = "bkg-peak"


class MapSelectionEnum(str, Enum):
    counts = "counts"
    exposure = "exposure"
    background = "background"
    psf = "psf"
    edisp = "edisp"


class GammapyBaseConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        validate_default=True,
        use_enum_values=True,
    )

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"


class SkyCoordConfig(GammapyBaseConfig):
    frame: Optional[FrameEnum] = None
    lon: Optional[AngleType] = None
    lat: Optional[AngleType] = None


class EnergyAxisConfig(GammapyBaseConfig):
    min: Optional[EnergyType] = None
    max: Optional[EnergyType] = None
    nbins: Optional[int] = None


class SpatialCircleConfig(GammapyBaseConfig):
    frame: Optional[FrameEnum] = None
    lon: Optional[AngleType] = None
    lat: Optional[AngleType] = None
    radius: Optional[AngleType] = None


class EnergyRangeConfig(GammapyBaseConfig):
    min: Optional[EnergyType] = None
    max: Optional[EnergyType] = None


class TimeRangeConfig(GammapyBaseConfig):
    start: Optional[TimeType] = None
    stop: Optional[TimeType] = None


class FluxPointsConfig(GammapyBaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class LightCurveConfig(GammapyBaseConfig):
    time_intervals: TimeRangeConfig = TimeRangeConfig()
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class FitConfig(GammapyBaseConfig):
    fit_range: EnergyRangeConfig = EnergyRangeConfig()


class ExcessMapConfig(GammapyBaseConfig):
    correlation_radius: AngleType = "0.1 deg"
    parameters: dict = {}
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()


class BackgroundConfig(GammapyBaseConfig):
    method: Optional[BackgroundMethodEnum] = None
    exclusion: Optional[PathType] = None
    parameters: dict = {}


class SafeMaskConfig(GammapyBaseConfig):
    methods: List[SafeMaskMethodsEnum] = [SafeMaskMethodsEnum.aeff_default]
    parameters: dict = {}


class EnergyAxesConfig(GammapyBaseConfig):
    energy: EnergyAxisConfig = EnergyAxisConfig(min="1 TeV", max="10 TeV", nbins=5)
    energy_true: EnergyAxisConfig = EnergyAxisConfig(
        min="0.5 TeV", max="20 TeV", nbins=16
    )


class SelectionConfig(GammapyBaseConfig):
    offset_max: AngleType = "2.5 deg"


class WidthConfig(GammapyBaseConfig):
    width: AngleType = "5 deg"
    height: AngleType = "5 deg"


class WcsConfig(GammapyBaseConfig):
    skydir: SkyCoordConfig = SkyCoordConfig()
    binsize: AngleType = "0.02 deg"
    width: WidthConfig = WidthConfig()
    binsize_irf: AngleType = "0.2 deg"


class GeomConfig(GammapyBaseConfig):
    wcs: WcsConfig = WcsConfig()
    selection: SelectionConfig = SelectionConfig()
    axes: EnergyAxesConfig = EnergyAxesConfig()


class DatasetsConfig(GammapyBaseConfig):
    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    stack: bool = True
    geom: GeomConfig = GeomConfig()
    map_selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = True


class ObservationsConfig(GammapyBaseConfig):
    datastore: PathType = Path("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids: List[int] = []
    obs_file: Optional[PathType] = None
    obs_cone: SpatialCircleConfig = SpatialCircleConfig()
    obs_time: TimeRangeConfig = TimeRangeConfig()
    required_irf: List[RequiredHDUEnum] = ["aeff", "edisp", "psf", "bkg"]


class LogConfig(GammapyBaseConfig):
    level: str = "info"
    filename: Optional[PathType] = None
    filemode: Optional[str] = None
    format: Optional[str] = None
    datefmt: Optional[str] = None


class GeneralConfig(GammapyBaseConfig):
    log: LogConfig = LogConfig()
    outdir: str = "."
    n_jobs: int = 1
    datasets_file: Optional[PathType] = None
    models_file: Optional[PathType] = None


class AnalysisConfig(GammapyBaseConfig):
    """Gammapy analysis configuration."""

    general: GeneralConfig = GeneralConfig()
    observations: ObservationsConfig = ObservationsConfig()
    datasets: DatasetsConfig = DatasetsConfig()
    fit: FitConfig = FitConfig()
    flux_points: FluxPointsConfig = FluxPointsConfig()
    excess_map: ExcessMapConfig = ExcessMapConfig()
    light_curve: LightCurveConfig = LightCurveConfig()

    def __str__(self):
        """Display settings in pretty YAML format."""
        info = self.__class__.__name__ + "\n\n\t"
        data = self.to_yaml()
        data = data.replace("\n", "\n\t")
        info += data
        return info.expandtabs(tabsize=4)

    @classmethod
    def read(cls, path):
        """Read from YAML file.

        Parameters
        ----------
        path : str
            input filepath
        """
        config = read_yaml(path)
        config.pop("metadata", None)
        return AnalysisConfig(**config)

    @classmethod
    def from_yaml(cls, config_str):
        """Create from YAML string.

        Parameters
        ----------
        config_str : str
            yaml str

        """
        settings = yaml.safe_load(config_str)
        return AnalysisConfig(**settings)

    def write(self, path, overwrite=False):
        """Write to YAML file.

        Parameters
        ----------
        path : `pathlib.Path` or str
            Path to write files.
        overwrite : bool, optional
            Overwrite existing file. Default is False.
        """
        yaml_str = self.to_yaml()
        write_yaml(yaml_str, path, overwrite=overwrite)

    def to_yaml(self):
        """Convert to YAML string."""
        data = json.loads(self.model_dump_json())
        return to_yaml(data)

    def set_logging(self):
        """Set logging config.

        Calls ``logging.basicConfig``, i.e. adjusts global logging state.
        """
        self.general.log.level = self.general.log.level.upper()
        logging.basicConfig(**self.general.log.model_dump())
        log.info("Setting logging config: {!r}".format(self.general.log.model_dump()))

    def update(self, config=None):
        """Update config with provided settings.

        Parameters
        ----------
        config : str or `AnalysisConfig` object, optional
            Configuration settings provided in dict() syntax. Default is None.
        """
        if isinstance(config, str):
            other = AnalysisConfig.from_yaml(config)
        elif isinstance(config, AnalysisConfig):
            other = config
        else:
            raise TypeError(f"Invalid type: {config}")

        config_new = deep_update(
            self.model_dump(exclude_defaults=True),
            other.model_dump(exclude_defaults=True),
        )
        return AnalysisConfig(**config_new)

    @staticmethod
    def _get_doc_sections():
        """Return dictionary with commented docs from docs file."""
        doc = defaultdict(str)
        with open(DOCS_FILE) as f:
            for line in filter(lambda line: not line.startswith("---"), f):
                line = line.strip("\n")
                if line.startswith("# Section: "):
                    keyword = line.replace("# Section: ", "")
                doc[keyword] += line + "\n"
        return doc
