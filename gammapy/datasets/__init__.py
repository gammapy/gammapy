from gammapy.utils.registry import Registry
from .core import *
from .map import *
from .spectrum import *
from .flux_points import *

DATASETS = Registry([MapDataset, SpectrumDatasetOnOff, FluxPointsDataset])
