r"""
.. _logparabola-spectral-model:

Log Parabola Spectral Model
===========================

This model is defined by the following equation:

.. math::
        \phi(E) = \phi_0 \left( \frac{E}{E_0} \right) ^ {
          - \alpha - \beta \log{ \left( \frac{E}{E_0} \right) }
        }

Note that :math:`log` refers to the natural logarithm. This is consistent
with the `Fermi Science Tools
<https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/source_models.html>`_
and `ctools
<http://cta.irap.omp.eu/ctools-devel/users/user_manual/getting_started/models.html#log-parabola>`_.
The `Sherpa <http://cxc.harvard.edu/sherpa/ahelp/logparabola.html_
package>`_ package, however, uses :math:`log_{10}`. If you have
parametrization based on :math:`log_{10}` you can use the
:func:`~gammapy.modeling.models.LogParabolaSpectralModel.from_log10` method.

"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

import matplotlib.pyplot as plt
from astropy import units as u
from gammapy.modeling.models import (
    Models,
    SkyModel,
    LogParabolaSpectralModel,
)

energy_range = [0.1, 100] * u.TeV
model = LogParabolaSpectralModel(
    alpha=2.3 * u.Unit(""),
    amplitude=4 / u.cm ** 2 / u.s / u.TeV,
    reference=1 * u.TeV,
    beta=0.5 * u.Unit(""),
)
model.plot(energy_range)
plt.grid(which="both");

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model)
models = Models([model])

print(models.to_yaml())
