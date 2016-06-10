### Using the Lightcurve class
from gammapy.time import LightCurve, make_example_lightcurve

lc = make_example_lightcurve()
print('the mean flux is {}'.format(lc['FLUX'].mean()))
print('the std dev of the flux is {}'.format(lc['FLUX'].std()))
lc.lc_plot()
