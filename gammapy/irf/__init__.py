# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Instrument response functions (IRFs).
"""
from gammapy.utils.registry import Registry
from .background import *
from .edisp_kernel import *
from .edisp_map import *
from .effective_area import *
from .energy_dispersion import *
from .io import *
from .psf_3d import *
from .psf_gauss import *
from .psf_kernel import *
from .psf_king import *
from .psf_map import *
from .psf_table import *

IRF_REGISTRY = Registry(
    [
        EffectiveAreaTable2D,
        EnergyDispersion2D,
        PSF3D,
        EnergyDependentMultiGaussPSF,
        PSFKing,
        Background3D,
        Background2D,
        PSFMap,
        EDispKernelMap,
    ]
)
