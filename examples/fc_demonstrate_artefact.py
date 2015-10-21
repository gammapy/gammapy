"""Demonstrate the artefact that can arise if
   fc_fix_limits is not used."""
import numpy as np
from astropy.table import Table
from gammapy.stats import fc_find_acceptance_interval_poisson

background = 3.5
cl = 0.90
x_bins = np.arange(0, 100)

table = Table()
table['mu'] = [0.745, 0.750, 0.755, 1.030, 1.035, 1.040, 1.045, 1.050, 1.055,
               1.060, 1.065]
table['x_min'] = 0.0
table['x_max'] = 0.0

for row in table:
    x_min, x_max = fc_find_acceptance_interval_poisson(row['mu'], background,
                                                       x_bins, cl)
    row['x_min'] = x_min
    row['x_max'] = x_max

table.pprint()
