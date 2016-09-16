"""
Using the Lightcurve class
"""
from gammapy.time import LightCurve

lc = LightCurve.simulate_example()
print('the mean flux is {}'.format(lc['FLUX'].mean()))
print('the std dev of the flux is {}'.format(lc['FLUX'].std()))
lc.lc_plot()
