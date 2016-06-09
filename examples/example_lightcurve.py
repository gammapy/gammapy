### Using the Lightcurve class
import astropy.units as u
from astropy.units import Quantity
from astropy.table import QTable
import gammapy.time
from gammapy.time.lightcurve import LightCurve
table = QTable()
table['TSTART'] = [1, 4, 7, 9] * u.s
table['TSTOP'] = [1, 4, 7, 9] * u.s
table['FLUX'] = Quantity([1, 4, 7, 9], 'cm^-2 s^-1')
table['ERRORS'] = Quantity([0.1, 0.4, 0.7, 0.9], 'cm^-2 s^-1')

lc = LightCurve(table=table)
print(lc.flux_mean())
lc.lc_plot()
