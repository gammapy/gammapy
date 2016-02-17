import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from astropy.tests.helper import assert_quantity_allclose
from astropy.table import Table
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.modeling.models import Gaussian1D
from gammapy.utils.testing import requires_dependency, requires_data
from gammapy.datasets import gammapy_extra
from gammapy.background import GaussianBand2D, CubeBackgroundModel, EnergyOffsetBackgroundModel
from gammapy.utils.energy import EnergyBounds, Energy
from gammapy.data import ObservationTable
from gammapy.data import DataStore
import pylab as pt
pt.ion()

def curve():
    dir = str(gammapy_extra.dir) + '/datasets/hess-crab4-hd-hap-prod2'
    data_store = DataStore.from_dir(dir)
    obs_table = data_store.obs_table

    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = Angle(np.linspace(0, 2.5, 100), "deg")
    multi_array = EnergyOffsetBackgroundModel(ebounds, offset)
    multi_array.fill_obs(obs_table, data_store)
    multi_array.compute_rate()

    bgarray=multi_array.bg_rate
    energ_range = Energy([1, 10], 'TeV')
    bins = 10
    table = bgarray.acceptance_curve_in_energy_band(energ_range, bins)

    pt.plot(table["offset"], table["Acceptance"])
    pt.xlabel("offset (deg)")
    pt.ylabel("Acceptance (s-1)")
    input()
    
def plot_array():
    dir = str(gammapy_extra.dir) + '/datasets/hess-crab4-hd-hap-prod2'
    data_store = DataStore.from_dir(dir)
    obs_table = data_store.obs_table

    ebounds = EnergyBounds.equal_log_spacing(0.1, 100, 100, 'TeV')
    offset = Angle(np.linspace(0, 2.5, 100), "deg")
    multi_array = EnergyOffsetBackgroundModel(ebounds, offset)
    multi_array.fill_obs(obs_table, data_store)
    multi_array.compute_rate()
    pt.figure(1)
    multi_array.counts.plot_image()
    pt.figure(2)
    multi_array.livetime.plot_image()
    pt.figure(3)
    multi_array.bg_rate.plot_image()
    input()
    
