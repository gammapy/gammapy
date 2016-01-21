import numpy as np
from gammapy.background import EnergyOffsetArray
from astropy.table import Table
from gammapy.data import DataStore
from gammapy.data import EventList
from gammapy.datasets import gammapy_extra
import pylab as pt

pt.ion()

# Create the Observation Table for the four Crab observations
dir = str(gammapy_extra.dir) + '/datasets/hess-crab4'
# Observation_Table=Table.read(dir+"/observations.fits.gz")
data_store = DataStore.from_dir(dir)
observation_table = data_store.obs_table

# Select the event fits files for these observations
event_list_files = data_store.make_table_of_files(observation_table,
                                                  filetypes=['events'])
# List of EventList Object for each observation
ev_list = []
for (i, i_ev_file) in enumerate(event_list_files['filename']):
    ev_list.append(EventList.read(i_ev_file))

# Define the bining in energy and offset
energy = np.logspace(-1, 2, 100)
offset = np.linspace(0, 2.5, 100)
array = EnergyOffsetArray(energy, offset)

# Fill the EnergyArray and plot the result
array.fill_events(ev_list)
array.plot_image()
