from .core import *
from .spectrum import *
from .map import *
from .flux_points import *
from .io import *
from gammapy.modeling.models import Registry

# TODO: move this elsewhere ?
DATASETS = Registry([MapDataset, SpectrumDatasetOnOff, FluxPointsDataset])
