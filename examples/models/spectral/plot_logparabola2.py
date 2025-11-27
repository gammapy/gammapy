r"""
.. _logparabola2-spectral-model:

Log parabola 2 spectral model
===========================

This model parametrises a log parabola spectrum,
where the energy scale of the exponente, :math:`E_s`, and the reference energy, :math:`E_0`, can be different.
It is defined by the following equation:

.. math::
        \phi(E) = \phi_0 \left( \frac{E}{E_0} \right) ^ {
          - \alpha - \beta \log{ \left( \frac{E}{E_s} \right) }
        }

Note that :math:`log` refers to the natural logarithm. 
If you have parametrization based on :math:`log_{10}` you can use the
:func:`~gammapy.modeling.models.LogParabola2SpectralModel.from_log10` method.

"""

# %%
# Example plot
# ------------
# Here is an example plot of the model:

from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.modeling.models import LogParabola2SpectralModel, Models, SkyModel

energy_bounds = [0.1, 100] * u.TeV
model = LogParabola2SpectralModel(
    alpha=2.3,
    amplitude="1e-12 cm-2 s-1 TeV-1",
    reference=10 * u.TeV,
    beta=0.5,
    escale=1 * u.TeV,
)
model.plot(energy_bounds)
plt.grid(which="both")

# %%
# YAML representation
# -------------------
# Here is an example YAML file using the model:

model = SkyModel(spectral_model=model, name="log-parabola2-model")
models = Models([model])

print(models.to_yaml())
