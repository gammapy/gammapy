r"""
.. _LightCurve-temporal-model:

LightCurve Temporal Model
=======================

This model parametrises a lightCurve time model.


"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from gammapy.modeling.models import Models, LightCurveTemplateTemporalModel, SkyModel

#start_time = Time("2015-10-01")
#end_time = Time("2020-10-01")
#norm_min = 0.01551196351647377
#norm_max = 1.0
#n1 = LightCurveTemplateTemporalModel._interpolator(end_time.mjd)
#n2 = LightCurveTemplateTemporalModel._interpolator(start_time.mjd)
#n_events = u.Quantity(n1 - n2, "day") / LightCurveTemplateTemporalModel.time_sum(start_time, end_time)
#t_delta="24 h"
#random_state=0
#time_range = [start_time * u.d , end_time * u.d]
#model = LightCurveTemplateTemporalModel(start_time = start_time.mjd * u.d, end_time=end_time * u.dn_events, t_delta= t_delta, random_state=random_state)
#model.plot(time_range)
#plt.grid(which="both")


import numpy as np
from gammapy.modeling.models import LightCurveTemplateTemporalModel
from astropy.time import Time
import astropy.units as u
time_range  = [Time("59100",format="mjd"), Time("59365",format="mjd")]
path = "$GAMMAPY_DATA/tests/models/light_curve/lightcrv_PKSB1222+216.fits"
light_curve_model = LightCurveTemplateTemporalModel.read(path)
light_curve_model.plot(time_range)