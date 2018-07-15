from gammapy.data import EventList
import numpy as np
from gammapy.time import Phasogram

events = EventList.read('../gammapy-extra/test_datasets/pulsed/hess_events_023523_phased.fits')
phase_bins = np.linspace(0, 1, 100)
phasogram = Phasogram.from_phase_bins(phase_bins)
phasogram.fill_events(events)
phasogram.table.pprint()

phasogram.plot()

import matplotlib.pyplot as plt
plt.show()

