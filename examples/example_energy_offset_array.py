import numpy as np
import matplotlib.pyplot as plt

from gammapy.data import DataStore
from gammapy.datasets import gammapy_extra
from gammapy.background import EnergyOffsetArray
from gammapy.utils.energy import EnergyBounds


def make_counts_array():
    """Make an example counts array with energy and offset axes."""
    data_store = DataStore.from_dir(gammapy_extra.dir / 'datasets/hess-crab4')

    event_lists = data_store.load_all('events')
    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = np.linspace(0, 2.5, 100)
    array = EnergyOffsetArray(ebounds, offset)
    array.fill_events(event_lists)

    return array


if __name__ == '__main__':
    array = make_counts_array()
    array.plot_image()
    plt.show()
