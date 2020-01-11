from .core import *
from .spectrum import *
from .map import *
from .flux_points import *
from .io import *
from gammapy.modeling.models import Registry

DATASETS = Registry([MapDataset, SpectrumDatasetOnOff, FluxPointsDataset])
