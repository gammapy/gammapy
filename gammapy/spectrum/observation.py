# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
from collections import UserList
from pathlib import Path
import numpy as np
from astropy.units import Quantity
from ..utils.scripts import make_path
from ..utils.energy import EnergyBounds
from ..utils.table import table_from_row_data
from ..data import ObservationStats
from ..irf import EffectiveAreaTable, EnergyDispersion, IRFStacker
from .core import CountsSpectrum, PHACountsSpectrum, PHACountsSpectrumList
from .utils import SpectrumEvaluator

__all__ = [
    "SpectrumStats",
]


class SpectrumStats(ObservationStats):
    """Spectrum stats.

    Extends `~gammapy.data.ObservationStats` with spectrum
    specific information (energy bin info at the moment).
    """

    def __init__(self, **kwargs):
        self.energy_min = kwargs.pop("energy_min", Quantity(0, "TeV"))
        self.energy_max = kwargs.pop("energy_max", Quantity(0, "TeV"))
        super().__init__(**kwargs)

    def __str__(self):
        ss = super().__str__()
        ss += "energy range: {:.2f} - {:.2f}".format(self.energy_min, self.energy_max)
        return ss

    def to_dict(self):
        """TODO: document"""
        data = super().to_dict()
        data["energy_min"] = self.energy_min
        data["energy_max"] = self.energy_max
        return data


