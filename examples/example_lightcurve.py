### Using the Lightcurve class
import astropy.units as u
from astropy.units import Quantity
from astropy.table import QTable
from gammapy.time import LightCurve
table = QTable()
table['TSTART'] = [1, 4, 7, 9] * u.s
table['TSTOP'] = [1, 4, 7, 9] * u.s
table['FLUX'] = Quantity([1, 4, 7, 9], 'cm^-2 s^-1')
table['ERRORS'] = Quantity([0.1, 0.4, 0.7, 0.9], 'cm^-2 s^-1')

lc = LightCurve(table=table)
print('the mean flux is {}'.format(lc.flux_mean()))
print('the std dev of the flux is {}'.format(lc.flux_std()))
lc.lc_plot()
