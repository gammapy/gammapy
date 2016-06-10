### Using the Lightcurve class
from gammapy.time import LightCurve, make_example_lightcurve

table = make_example_lightcurve()
lc = LightCurve(table=table)
print('the mean flux is {}'.format(lc.flux_mean()))
print('the std dev of the flux is {}'.format(lc.flux_std()))
lc.lc_plot()
