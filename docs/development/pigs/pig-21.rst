.. include:: ../../references.txt

.. _pig-021:

****************************
PIG 21 - Models Improvements
****************************

* Author: Axel Donath, Régis Terrier and Quentin Remy
* Created: Jun 10, 2020
* Accepted:
* Status:
* Discussion:

Abstract
========
This PIG outlines further improvement to the modeling framework in Gammapy.


Proposal
========

Spectral Norm Models
--------------------

For the purpose of adjusting template based models we propose to introduce a new
class of spectral models. Those spectral models feature a norm parameter instead
of amplitude and are named using the ``NormSpectralModel`` suffix:

.. code::

    from gammapy.modeling.models import PowerLawNormSpectralModel, LogParabolaNormSpectralModel, NodeNormSpectralModel

    pwl_norm = PowerLawNormSpectralModel()
    log_parabola_norm = LogParabolaNormSpectralModel()
    bpwl_norm = PiecewiseBrokenPowerlawNormSpectralModel()
    const_norm = ConstantNormSpectralModel()


Energy Dependent Spatial Models
-------------------------------
A very common use-case in scientific analyses is to look for energy dependent
morphology of extended sources. In the past this kind of analysis has been typically
done by splitting the data into energy bands and doing individual fits of the
morphology in these energy bands. In a combined spectral and spatial ("cube") analysis
this can be naturally achieved by allowing for an energy dependent spatial model,
where the energy dependence is e.g. described by a parametric model expression with
energy dependent parameters.

In the current model scheme we use a "factorised" representation of the source model,
where the spatial, energy and time dependence are independent model components and
not correlated:

.. math::

    f(l, b, E, t) = A \cdot F(l, b) \cdot G(E) \cdot H(t)

To represent energy dependent morphology we propose to introduce energy
dependent spatial models:

.. math::

    f(l, b, E, t) = A \cdot F(l, b, E) \cdot G(E) \cdot H(t)

In general the energy dependence is optional. If the spatial model does not declare
an energy dependence it assumes the same morphology for all energies. This also ensures backwards
compatibility with the current behaviour.

To limit the implementation effort in this PIG we propose to only adapt the ``SkyDiffuseCube``
and add an example energy dependent custom model to our documentation. We do not propose to
introduce general dependence of arbitrary model parameters for any spatial model, such as
``GaussianSpatialModel`` or ``DiskSpatialModel``. An example of how this can be achieved with
a custom implemented model is given below.

We propose to add energy dependence to the ``TemplateSpatialModel`` and
replace the current ``SkyDiffuseCube`` by:


.. code::

    spatial_model = TemplateSpatialModel.read("my_cube.fits")

    model = SkyModel(
        spatial_model=spatial_model,
        spectral_model=PowerLawNormSpectralModel()
    )

A custom energy dependent spatial model can be implemented like:


.. code::

    from gammapy.modeling.models import SpatialModel
    from astropy.coordinates.angle_utilities import angular_separation

    class MyCustomGaussianModel(SpatialModel):
        """My custom gaussian model.

        Parameters
        ----------
        lon_0, lat_0 : `~astropy.coordinates.Angle`
            Center position
        sigma_1TeV : `~astropy.coordinates.Angle`
            Width of the Gaussian at 1 TeV
        sigma_10TeV : `~astropy.coordinates.Angle`
            Width of the Gaussian at 10 TeV

        """
        tag = "MyCustomGaussianModel"
        lon_0 = Parameter("lon_0", "0 deg")
        lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)

        sigma_1TeV = Parameter("sigma_1TeV", "1 deg", min=0)
        sigma_10TeV = Parameter("sigma_10TeV", "0.5 deg", min=0)

    @staticmethod:
    def evaluate(lon, lat, energy, lon_0, lat_0, sigma_1TeV, sigma_10TeV):
        """Evaluate custom Gaussian model"""
        sigmas = u.Quantity([sigma_1TeV, sigma_10TeV])
        energy_nodes = [1, 10] * u.TeV
        sigma = np.interp(energy, energy_nodes, sigmas)

        sep = angular_separation(lon, lat, lon_0, lat_0)

        exponent = -0.5 * (sep / sigma) ** 2
        norm = 1 / (2 * np.pi * sigma ** 2)
        return norm * np.exp(exponent)

    @property
    def evaluation_radius(self):
        """Evaluation radius (`~astropy.coordinates.Angle`)."""
        return 5 * self.sigma_1TeV.quantity



Spectral Absorption Model
-------------------------

In the current handling of absorbed spectral models we have a very special
``Absorption`` model, which is not a spectral model. To resolve this special
case, we propose to refactor the existing code and handle the absorbed case
using a ``CompoundSpectralModel``. The new implementation is used as follows:

.. code::

    from gammapy.modeling.models import EBLAbsorptionNormSpectralModel, PowerLawSpectralModel

    absorption = EBLAbsorptionNormSpectralModel.from_reference(
        redshift=0.1, alpha_norm=1, reference="dominguez"
    )

    pwl = PowerLawSpectralModel()

    spectral_model = absorption * pwl

    assert isinstance(spectral_model, CompoundSpectralModel)

In addition we propose to rename ``.table_model`` to ``.to_template_spectral_model(redshift, alpha_norm)``.


XML Support for Reading Models
------------------------------
ctools as well as the Fermi Science Tools use a XML-based model serialisation
format. To ensure compatibility with these tools we propose to add
support for reading XML files to Gammapy, so that the following works:

.. code::

    from gammapy.modeling.models import Models

    models = Models.read("my_model.xml")
    print(models)


**Alternatives:** Alternatively one could implement a conversion script, that
allows to convert a model file from YAML to XML format. This script would be
shipped with Gammapy and be available as a sub-command ``gammapy convert-model-file my_model.xml my_model.yaml``.
The effort of implementation is comparable.

For complete support it is necessary to add further models to Gammapy,
which are listed in the following section.


Additional Models
-----------------
In addition we propose to implement the following models in ``gammapy.modeling.models``:


- ``SersicSpatialModel`` following the parametrisation of the Astropy ``Sersic2D`` model.

- ``BrokenPowerLaw`` with the following parametrisation:

.. math::

    \begin{split}\phi(E) = \phi_0 \times \left \{
	\begin{eqnarray}
  		\left( \frac{E}{E_b} \right)^{\gamma_1} & {\rm if\,\,} E < E_b \\
  		\left( \frac{E}{E_b} \right)^{\gamma_2} & {\rm otherwise}
	\end{eqnarray}
	\right .\end{split}



- ``PiecewiseBrokenPowerLawSpectralModel``

.. code::

    energy = [1, 10, 100] * u.TeV
    amplitudes = [1e-12, 1e-13, 1e-15] * u.Unit("TeV-1 cm-2 s-1")

    model = PiecewiseBrokenPowerLawSpectralModel(
        energy=energy, amplitudes=amplitudes
    )


    print(model.amplitude_0)
    print(model.amplitude_1)


- Reintroduce the ``PhaseCurveModel`` to compute mean fluxes over time


Simplify YAML Representation
----------------------------

Introduce Shorter YAML Alias Tags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To simplify the definition of models in YAML files as well as for interactive
model creation we propose to introduce shorter YAML tags for all models in Gammapy.
For backwards compatibility we propose to support the class name as well. We
require that the short YAML tag is only unique within the model class, so that
the same tag such as "gauss" can be re-used by spectral, spatial as well as
temporal components. The uniqueness is guaranteed in the YAML file, because
of the different sections for the model types. For writing the models the
verbose class name is used by default.

We propose to introduce the following YAML tags:

======================================== ======================
Class Name                               YAML Tag
======================================== ======================
ConstantSpectralModel                    const
CompoundSpectralModel                    compound
PowerLawSpectralModel                    pl
PowerLawNormSpectralModel                pl-norm
PowerLaw2SpectralModel                   pl-2
SmoothBrokenPowerLawSpectralModel        sbpl
BrokenPowerLawSpectralModel				 bpl
PiecewiseBrokenPowerLawNormSpectraModel  pwbpl-norm
ExpCutoffPowerLawSpectralModel           ecpl
ExpCutoffPowerLaw3FGLSpectralModel       ecpl-3fgl
SuperExpCutoffPowerLaw3FGLSpectralModel  secpl-3fgl
SuperExpCutoffPowerLaw4FGLSpectralModel  secpl-4fgl
LogParabolaSpectralModel                 lp
LogParabolaNormSpectralModel             lp-norm
TemplateSpectralModel                    template
GaussianSpectralModel                    gauss
EBLAbsorbtionNormSpectralModel           ebl-absorbtion-norm
NaimaSpectralModel                       naima
ScaleSpectralModel                       scale
======================================== ======================


======================================== ======================
Class Name                               YAML Tag
======================================== ======================
ConstantSpatialModel                     const
TemplateSpatialModel                     template
GaussianSpatialModel                     gauss
DiskSpatialModel    					 disk
PointSpatialModel						 point
ShellSpatialModel                        shell
SersicSpatialModel						 sersic
======================================== ======================


======================================== ======================
Class Name                               YAML Tag
======================================== ======================
ConstantTemporalModel          			 const
LightCurveTemplateTemporalModel          template
GaussianTemporalModel					 gauss
ExpDecayTemporalModel					 exp-decay
======================================== ======================

To simplify the interactive model creation we propose to introduce:

.. code::

    from gammapy.modeling.models import SkyModel

    model = SkyModel.create(
    	spectral_type="pl",
		spectral_pars={"index": 2},
		spatial_type="gauss",
		spatial_pars={"sigma": "0.1 deg"},
		temporal_type="const",
    )

In addition the ``Model.create()`` factory function should be
adapted to:

.. code::

    from gammapy.modeling.models import Model

    pwl = Model.create(tag="gauss", model_type="spectral", **pars)


Simplify YAML Parameter Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To further simplify the structure of the YAML file we propose to remove parameter
properties if they are equivalent to the default values. The current representation
looks like:

.. code::

    spectral:
        type: PowerLawSpectralModel
        parameters:
        - name: index
          value: 2.0
          unit: ''
          min: .nan
          max: .nan
          frozen: false
          error: 0
        - name: amplitude
          value: 1.0e-12
          unit: cm-2 s-1 TeV-1
          min: .nan
          max: .nan
          frozen: false
          error: 0
        - name: reference
          value: 1.0
          unit: TeV
          min: .nan
          max: .nan
          frozen: true
          error: 0

After the simplification:

.. code::

    spectral:
        type: PowerLawSpectralModel
        parameters:
        - name: index
          value: 2.0
          unit: ''
        - name: amplitude
          value: 1.0e-12
          unit: cm-2 s-1 TeV-1
        - name: reference
          value: 1.0
          unit: TeV
          frozen: true


Decision
========

.. _gammapy: https://github.com/gammapy/gammapy
.. _gammapy-web: https://github.com/gammapy/gammapy-webpage


