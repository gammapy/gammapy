"""Example how to use `gammapy.background.EnergyOffsetArray`.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from gammapy.data import DataStore
from gammapy.background import EnergyOffsetArray
from gammapy.utils.energy import EnergyBounds


def make_counts_array():
    """Make an example counts array with energy and offset axes."""
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2')

    event_lists = data_store.load_all('events')
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = Angle(np.linspace(0, 2.5, 100), "deg")
    array = EnergyOffsetArray(ebounds, offset)
    array.fill_events(event_lists)

    return array


if __name__ == '__main__':
    array = make_counts_array()
    array.plot()
    plt.show()
