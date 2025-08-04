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
    """Base configuration class.

    Provides Pydantic model settings.
    """

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
    """Configuration for `~astropy.coordinates.SkyCoord`.

    Attributes
    ----------
    frame : str
        Coordinate frame (e.g., 'icrs', 'galactic').
    lon : `astropy.units.Quantity`
        Longitude or right ascension.
    lat : `astropy.units.Quantity`
        Latitude or declination.
    """

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
    """Configuration for the `~gammapy.estimators.FluxPointsEstimator`.

    Attributes
    ----------
    energy : dict
        Energy binning for the light curve. Should contain the following keys
        'min' and 'max' (with energy quantities) and 'nbins'.
    source : str
        Source name.
    parameters : dict
        Optional parameters such as selection filters, options are:
        "all", "errn-errp", "ul", "scan"
    """

    energy: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class LightCurveConfig(GammapyBaseConfig):
    """Configuration for the `~gammapy.estimators.LightCurveEstimator`.

    Attributes
    ----------
    time_intervals : dict
        Time intervals for the light curve. Should contain the following keys
        'start' and 'stop' as time quantities.
    energy_edges : dict
        Energy binning for the light curve. Should contain the following keys
        'min' and 'max' (with energy quantities) and 'nbins'.
    source : str
        Source name.
    parameters : dict
        Optional parameters such as selection filters.
    """

    time_intervals: TimeRangeConfig = TimeRangeConfig()
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()
    source: str = "source"
    parameters: dict = {"selection_optional": "all"}


class FitConfig(GammapyBaseConfig):
    """Configuration for model fitting.

    Attributes
    ----------
    fit_range : dict
        Energy range used during the fit. Should contain the following keys
        'min' and 'max' (with energy quantities).
    """

    fit_range: EnergyRangeConfig = EnergyRangeConfig()


class ExcessMapConfig(GammapyBaseConfig):
    """Configuration for the `~gammapy.estimators.ExcessMapEstimator`.

    Attributes
    ----------
    correlation_radius : `astropy.units.Quantity`
        Radius for correlation/smoothing.
    parameters : dict
        Optional configuration parameters.
    energy_edges : dict
        Energy binning for map creation. Should contain the following keys
        'min' and 'max' (with energy quantities) and 'nbins'.
    """

    correlation_radius: AngleType = "0.1 deg"
    parameters: dict = {}
    energy_edges: EnergyAxisConfig = EnergyAxisConfig()


class BackgroundConfig(GammapyBaseConfig):
    """Configuration for background modeling.

    Attributes
    ----------
    method : list of str
        Background estimation method, the available options are:
        "reflected", "fov_background", "ring"
    exclusion : str
        Path to an exclusion region file.
    parameters : dict
        Additional options or method parameters.
    """

    method: Optional[BackgroundMethodEnum] = None
    exclusion: Optional[PathType] = None
    parameters: dict = {}


class SafeMaskConfig(GammapyBaseConfig):
    """Configuration for `~gammapy.makers.SafeMaskMaker`.

    Attributes
    ----------
    methods : list of str
        Masking methods to apply, the available options are:
        "aeff-default", "aeff-max", "edisp-bias", "offset-max", "bkg-peak"
    parameters : dict
        Additional configuration parameters.
    """

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
    """Configuration for the WCS geometry.

    Attributes
    ----------
    skydir : `SkyCoordConfig`
        Sky coordinates configuration.
    binsize : `astropy.units.Quantity`
        Pixel size in degrees.
    width : dict
        Spatial extent of the map. Should contain one or both of the keys
        'width' and 'height', with angular quantities as values.
    binsize_irf : `astropy.units.Quantity`
        Pixel size for IRF-related maps.
    """

    skydir: SkyCoordConfig = SkyCoordConfig()
    binsize: AngleType = "0.02 deg"
    width: WidthConfig = WidthConfig()
    binsize_irf: AngleType = "0.2 deg"


class GeomConfig(GammapyBaseConfig):
    """Configuration for geometry.

    Attributes
    ----------
    wcs : `WcsConfig`
        WCS geometry configuration.
    selection : `astropy.units.Quantity`
        The only option here is 'offset_max`.
    axes : `EnergyAxesConfig`
        Configuration of energy axes.
    """

    wcs: WcsConfig = WcsConfig()
    selection: SelectionConfig = SelectionConfig()
    axes: EnergyAxesConfig = EnergyAxesConfig()


class DatasetsConfig(GammapyBaseConfig):
    """Configuration for dataset reduction.

    Attributes
    ----------
    type : str
        Type of dataset to create (e.g., 'spectrum').
    stack : bool
        Whether to stack observations.
    geom : `GeomConfig`
        Geometry configuration.
    map_selection : list of str
        Select which maps to make, the available options are:
        'counts', 'exposure', 'background', 'psf', 'edisp'.
    background : `BackgroundConfig`
        Background configuration.
    safe_mask : `SafeMaskConfig`
        Safe mask configuration.
    on_region : dict
        ON-region definition for spectral extraction. Should contain the following keys
        'frame', 'lat', 'lon' and 'radius', the latter three as angle quantities.
    containment_correction : bool
        Whether to apply containment correction.
    """

    type: ReductionTypeEnum = ReductionTypeEnum.spectrum
    stack: bool = True
    geom: GeomConfig = GeomConfig()
    map_selection: List[MapSelectionEnum] = MapDatasetMaker.available_selection
    background: BackgroundConfig = BackgroundConfig()
    safe_mask: SafeMaskConfig = SafeMaskConfig()
    on_region: SpatialCircleConfig = SpatialCircleConfig()
    containment_correction: bool = True


class ObservationsConfig(GammapyBaseConfig):
    """Configuration for `~gammapy.data.Observations`.

    Attributes
    ----------
    datastore : str
        Path to the data store.
    obs_ids : list of int
        List of observation IDs.
    obs_file : str
        Path to a YAML observation file.
    obs_cone : dict
        Cone selection for observations. Should contain the following keys
        'frame', 'lat', 'lon' and 'radius', the latter three as angle quantities.
    obs_time : dict
        Observation time filtering.  Should contain the following keys
        'start' and 'stop' as time quantities.
    required_irf : list
        Required IRF components, options are "aeff", "edisp", "psf", "bkg"
    """

    datastore: PathType = Path("$GAMMAPY_DATA/hess-dl3-dr1/")
    obs_ids: List[int] = []
    obs_file: Optional[PathType] = None
    obs_cone: SpatialCircleConfig = SpatialCircleConfig()
    obs_time: TimeRangeConfig = TimeRangeConfig()
    required_irf: List[RequiredHDUEnum] = ["aeff", "edisp", "psf", "bkg"]


class LogConfig(GammapyBaseConfig):
    """Configuration for logging.

    Attributes
    ----------
    level : str
        Logging level (e.g., 'info', 'debug').
    filename : str
        Log file path.
    filemode : str
        File mode ('w' for overwrite, 'a' for append).
    format : str
        Logging format string.
    datefmt : str
        Format for timestamps.
    """

    level: str = "info"
    filename: Optional[PathType] = None
    filemode: Optional[str] = None
    format: Optional[str] = None
    datefmt: Optional[str] = None


class GeneralConfig(GammapyBaseConfig):
    """Top-level general configuration.

    Attributes
    ----------
    log : `LogConfig`
        Logging configuration.
    outdir : str
        Output directory.
    n_jobs : int
        Number of parallel jobs.
    datasets_file : str
        Path to datasets config file.
    models_file : str
        Path to models config file.
    """

    log: LogConfig = LogConfig()
    outdir: str = "."
    n_jobs: int = 1
    datasets_file: Optional[PathType] = None
    models_file: Optional[PathType] = None


class AnalysisConfig(GammapyBaseConfig):
    """Gammapy analysis configuration.

    This class defines the full analysis configuration schema, organised into different sections.
    It can be read from or written to a YAML file using `.read()` and `.write()`, respectively.

    Attributes
    ----------
    general : `GeneralConfig`
        General settings for output, logging, and file paths.
    observations : `ObservationsConfig`
        Settings for the `~gammapy.data.Observation` selection including IDs, regions, and time filters.
    datasets : `DatasetsConfig`
        Settings for the `~gammapy.datasets.Dataset` Including but not limited to geometry (`GeomConfig`),
        background (`BackgroundConfig`), safe mask (`SafeMaskConfig`), and stacking.
    fit : `FitConfig`
        Configuration for the `~gammapy.modeling.Fit` strategy and global fit energy range.
    flux_points : `FluxPointsConfig`
        Configuration for the `~gammapy.estimators.FluxPointsEstimator`.
    excess_map : `ExcessMapConfig`
        Configuration for the `~gammapy.estimators.ExcessMapEstimator`.
    light_curve : `LightCurveConfig`
        Configuration for the `~gammapy.estimators.LightCurveEstimator`.

    Examples
    --------
    Read from a yaml file::

    >>> from gammapy.analysis import AnalysisConfig
    >>> config = AnalysisConfig.read("config.yaml") # doctest: +SKIP
    >>> print(config.datasets.geom) # doctest: +SKIP

    Create from scratch

    >>> config = AnalysisConfig()
    >>> config.observations.datastore = "$GAMMAPY_DATA/hess-dl3-dr1"
    >>> config.observations.obs_cone = {"frame": "icrs", "lon": "83.633 deg", "lat": "22.014 deg", "radius": "5 deg"}
    >>> print(config.observations.obs_cone.lat.deg)
    22.014
    """

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
