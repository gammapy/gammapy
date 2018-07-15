# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.table import Table

__all__ = [
    'Phasogram',
]


class Phasogram:
    def __init__(self, table):
        self.table = table

    @classmethod
    def from_phase_bins(cls, phase_bins):
        table = Table()
        table['PHASE_MIN'] = phase_bins[:-1]
        table['PHASE_MAX'] = phase_bins[1:]
        return cls(table=table)

    @property
    def phase_bins(self):
        phase_min = self.table['PHASE_MIN']
        phase_max = self.table['PHASE_MAX']
        return np.concatenate([phase_min, [phase_max[-1]]])

    def fill_events(self, events):
        phase = events.table['PHASE']
        bins = self.phase_bins
        value = np.histogram(phase, bins=bins)[0]
        self.table['VALUE'] = value

    def plot(self):
        import matplotlib.pyplot as plt
        x = self.table['PHASE_MIN']
        y = self.table['VALUE']
        plt.bar(x, y)
