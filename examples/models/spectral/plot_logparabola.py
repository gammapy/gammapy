r"""
.. _logparabola-spectral-model:

Log Parabola Spectral Model
===========================

This model parametrises a log parabola spectrum.

It is defined by the following equation:

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

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import LogParabolaSpectralModel, Models, SkyModel

energy_range = [0.1, 100] * u.TeV
model = LogParabolaSpectralModel(
    alpha=2.3, amplitude="1e-12 cm-2 s-1 TeV-1", reference=1 * u.TeV, beta=0.5,
)
model.plot(energy_range)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="log-parabola-model")
models = Models([model])

print(models.to_yaml())
