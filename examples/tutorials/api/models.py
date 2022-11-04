"""
Models
======
This is an introduction and overview on how to work with models in
Gammapy.

The sub-package `~gammapy.modeling` contains all the functionality
related to modeling and fitting data. This includes spectral, spatial
and temporal model classes, as well as the fit and parameter API.The
models follow a naming scheme which contains the category as a suffix to
the class name. An overview of all the available models can be found in
the `model gallery <../../user-guide/model-gallery/index.rst>`__.

Note that there are separate tutorials,
:doc:`/tutorials/api/model_management` and
:doc:`/tutorials/api/fitting` that explains about
`~gammapy.modeling`, the Gammapy modeling and fitting framework. You
have to read that to learn how to work with models in order to analyse
data.

"""


######################################################################
# Setup
# -----
#

# %matplotlib inline
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from gammapy.maps import Map, MapAxis, WcsGeom

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# Spectral models
# ---------------
#
# All models are imported from the `~gammapy.modeling.models` namespace.
# Let’s start with a `PowerLawSpectralModel`:
#

from gammapy.modeling.models import PowerLawSpectralModel

pwl = PowerLawSpectralModel()
print(pwl)


######################################################################
# To get a list of all available spectral models you can import and print
# the spectral model registry or take a look at the :ref:`spectral-models-gallery`
#

from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

print(SPECTRAL_MODEL_REGISTRY)


######################################################################
# Spectral models all come with default parameters. Different parameter
# values can be passed on creation of the model, either as a string
# defining the value and unit or as an `astropy.units.Quantity` object
# directly:
#

amplitude = 1e-12 * u.Unit("TeV-1 cm-2 s-1")
pwl = PowerLawSpectralModel(amplitude=amplitude, index=2.2)


######################################################################
# For convenience a `str` specifying the value and unit can be passed as
# well:
#

pwl = PowerLawSpectralModel(amplitude="2.7e-12 TeV-1 cm-2 s-1", index=2.2)
print(pwl)


######################################################################
# The model can be evaluated at given energies by calling the model
# instance:
#

energy = [1, 3, 10, 30] * u.TeV
dnde = pwl(energy)
print(dnde)


######################################################################
# The returned quantity is a differential photon flux.
#
# For spectral models you can additionally compute the integrated and
# energy flux in a given energy range:
#

flux = pwl.integral(energy_min=1 * u.TeV, energy_max=10 * u.TeV)
print(flux)

eflux = pwl.energy_flux(energy_min=1 * u.TeV, energy_max=10 * u.TeV)
print(eflux)


######################################################################
# This also works for a list or an array of integration boundaries:
#

energy = [1, 3, 10, 30] * u.TeV
flux = pwl.integral(energy_min=energy[:-1], energy_max=energy[1:])
print(flux)


######################################################################
# In some cases it can be useful to find use the inverse of a spectral
# model, to find the energy at which a given flux is reached:
#

dnde = 2.7e-12 * u.Unit("TeV-1 cm-2 s-1")
energy = pwl.inverse(dnde)
print(energy)


######################################################################
# As a convenience you can also plot any spectral model in a given energy
# range:
#

plt.figure()
pwl.plot(energy_bounds=[1, 100] * u.TeV)


######################################################################
# Norm Spectral Models
# ~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Normed spectral models are a special class of Spectral Models, which
# have a dimension-less normalisation. These spectral models feature a
# norm parameter instead of amplitude and are named using the
# `NormSpectralModel` suffix. They **must** be used along with another
# spectral model, as a multiplicative correction factor according to their
# spectral shape. They can be typically used for adjusting template based
# models, or adding a EBL correction to some analytic model.
#
# To check if a given `SpectralModel` is a norm model, you can simply
# look at the `is_norm_spectral_model` property
#

# To see the available norm models shipped with gammapy:
for model in SPECTRAL_MODEL_REGISTRY:
    if model.is_norm_spectral_model:
        print(model)


######################################################################
# As an example, we see the `PowerLawNormSpectralModel`
#

from gammapy.modeling.models import PowerLawNormSpectralModel

pwl_norm = PowerLawNormSpectralModel(tilt=0.1)
print(pwl_norm)


######################################################################
# We can check the correction introduced at each energy
#

energy = [0.3, 1, 3, 10, 30] * u.TeV
print(pwl_norm(energy))


######################################################################
# A typical use case of a norm model would be in applying spectral
# correction to a `TemplateSpectralModel`. A template model is defined
# by custom tabular values provided at initialization.
#

from gammapy.modeling.models import TemplateSpectralModel

plt.figure()
energy = [0.3, 1, 3, 10, 30] * u.TeV
values = [40, 30, 20, 10, 1] * u.Unit("TeV-1 s-1 cm-2")
template = TemplateSpectralModel(energy, values)
template.plot(energy_bounds=[0.2, 50] * u.TeV, label="template model")
normed_template = template * pwl_norm
normed_template.plot(energy_bounds=[0.2, 50] * u.TeV, label="normed_template model")
plt.legend()


######################################################################
# Compound Spectral Model
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# A `CompoundSpectralModel` is an arithmetic combination of two spectral
# models. The model `normed_template` created in the preceding example
# is an example of a `CompoundSpectralModel`
#

print(normed_template)


######################################################################
# To create an additive model, you can do simply:
#

model_add = pwl + template
print(model_add)


######################################################################
# Spatial models
# --------------
#


######################################################################
# Spatial models are imported from the same `~gammapy.modeling.models`
# namespace, let’s start with a `GaussianSpatialModel`:
#

from gammapy.modeling.models import GaussianSpatialModel

gauss = GaussianSpatialModel(lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg")
print(gauss)


######################################################################
# Again you can check the `SPATIAL_MODELS` registry to see which models
# are available or take a look at the :ref:`spatial-models-gallery`
#

from gammapy.modeling.models import SPATIAL_MODEL_REGISTRY

print(SPATIAL_MODEL_REGISTRY)


######################################################################
# The default coordinate frame for all spatial models is `"icrs"`, but
# the frame can be modified using the `frame` argument:
#

gauss = GaussianSpatialModel(
    lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic"
)


######################################################################
# You can specify any valid `astropy.coordinates` frame. The center
# position of the model can be retrieved as a
# `astropy.coordinates.SkyCoord` object using `SpatialModel.position`:
#

print(gauss.position)


######################################################################
# Spatial models can be evaluated again by calling the instance:
#

lon = [0, 0.1] * u.deg
lat = [0, 0.1] * u.deg

flux_per_omega = gauss(lon, lat)
print(flux_per_omega)


######################################################################
# The returned quantity corresponds to a surface brightness. Spatial model
# can be also evaluated using `~gammapy.maps.Map` and
# `~gammapy.maps.Geom` objects:
#

m = Map.create(skydir=(0, 0), width=(1, 1), binsz=0.02, frame="galactic")
m.quantity = gauss.evaluate_geom(m.geom)
plt.figure()
m.plot(add_cbar=True)


######################################################################
# Again for convenience the model can be plotted directly:
#
plt.figure()
gauss.plot(add_cbar=True)


######################################################################
# All spatial models have an associated sky region to it e.g. to
# illustrate the extend of the model on a sky image. The returned object
# is an `~regions.SkyRegion` object:
#

print(gauss.to_region())


######################################################################
# Now we can plot the region on an sky image:
#

# create and plot the model
plt.figure()
gauss_elongated = GaussianSpatialModel(
    lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", e=0.7, phi="45 deg"
)
ax = gauss_elongated.plot(add_cbar=True)

# add region illustration
region = gauss_elongated.to_region()
region_pix = region.to_pixel(ax.wcs)
ax.add_artist(region_pix.as_artist(ec="w", fc="None"))


######################################################################
# The `~gammapy.modeling.models.SpatialModel.to_region()` method can also be useful to write e.g. ds9 region
# files using `write_ds9` from the `regions` package:
#

from regions import Regions

regions = Regions([gauss.to_region(), gauss_elongated.to_region()])

filename = "regions.reg"
regions.write(
    filename,
    format="ds9",
    overwrite=True,
)

# !cat regions.reg


######################################################################
# Temporal models
# ---------------
#


######################################################################
# Temporal models are imported from the same `~gammapy.modeling.models`
# namespace, let’s start with a `GaussianTemporalModel`:
#

from gammapy.modeling.models import GaussianTemporalModel

gauss_temp = GaussianTemporalModel(t_ref=59240.0 * u.d, sigma=2.0 * u.d)
print(gauss_temp)


######################################################################
# To check the `TEMPORAL_MODELS` registry to see which models are
# available:
#

from gammapy.modeling.models import TEMPORAL_MODEL_REGISTRY

print(TEMPORAL_MODEL_REGISTRY)


######################################################################
# Temporal models can be evaluated on `astropy.time.Time` objects. The
# returned quantity is a dimensionless number
#

from astropy.time import Time

time = Time("2021-01-29 00:00:00.000")
gauss_temp(time)


######################################################################
# As for other models, they can be plotted in a given time range
#

time = Time([59233.0, 59250], format="mjd")
gauss_temp.plot(time)


######################################################################
# SkyModel
# --------
#


######################################################################
# The `~gammapy.modeling.models.SkyModel` class combines a spectral, and
# optionally, a spatial model and a temporal. It can be created from
# existing spectral, spatial and temporal model components:
#

from gammapy.modeling.models import SkyModel

model = SkyModel(
    spectral_model=pwl,
    spatial_model=gauss,
    temporal_model=gauss_temp,
    name="my-source",
)
print(model)


######################################################################
# It is good practice to specify a name for your sky model, so that you
# can access it later by name and have meaningful identifier you
# serialisation. If you don’t define a name, a unique random name is
# generated:
#

model_without_name = SkyModel(spectral_model=pwl, spatial_model=gauss)
print(model_without_name.name)


######################################################################
# The individual components of the source model can be accessed using
# ``.spectral_model``, ``.spatial_model`` and ``.temporal_model``:
#

print(model.spectral_model)

print(model.spatial_model)

print(model.temporal_model)


######################################################################
# And can be used as you have seen already seen above:
#

plt.figure()
model.spectral_model.plot(energy_bounds=[1, 10] * u.TeV)


######################################################################
# Note that the gammapy fitting can interface only with a `~gammapy.modeling.models.SkyModel` and
# **not** its individual components. So, it is customary to work with
# `~gammapy.modeling.models.SkyModel` even if you are not doing a 3D fit. Since the amplitude
# parameter resides on the `~gammapy.modeling.models.SpectralModel`, specifying a spectral
# component is compulsory. The temporal and spatial components are
# optional. The temporal model needs to be specified only for timing
# analysis. In some cases (e.g. when doing a spectral analysis) there is
# no need for a spatial component either, and only a spectral model is
# associated with the source.
#

model_spectrum = SkyModel(spectral_model=pwl, name="source-spectrum")
print(model_spectrum)


######################################################################
# Additionally the spatial model of `~gammapy.modeling.models.SkyModel`
# can be used to represent source models based on templates, where the
# spatial and energy axes are correlated. It can be created e.g. from an
# existing FITS file:
#

from gammapy.modeling.models import PowerLawNormSpectralModel, TemplateSpatialModel

diffuse_cube = TemplateSpatialModel.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz", normalize=False
)
diffuse = SkyModel(PowerLawNormSpectralModel(), diffuse_cube)
print(diffuse)


######################################################################
# Note that if the spatial model is not normalized over the sky it has to
# be combined with a normalized spectral model, for example
# `~gammapy.modeling.models.PowerLawNormSpectralModel`. This is the only
# case in `gammapy.models.SkyModel` where the unit is fully attached to
# the spatial model.
#


######################################################################
# Modifying model parameters
# --------------------------
#
# Model parameters can be modified (eg: frozen, values changed, etc at any
# point), eg:
#

# Freezing a parameter
model.spectral_model.index.frozen = True
# Making a parameter free
model.spectral_model.index.frozen = False

# Changing a value
model.spectral_model.index.value = 3

# Setting min and max ranges on parameters
model.spectral_model.index.min = 1.0
model.spectral_model.index.max = 5.0

# Visualise the model as a table
model.parameters.to_table().show_in_notebook()


######################################################################
# You can use the interactive boxes to choose model parameters by name,
# type or other attrributes mentioned in the column names.
#


######################################################################
# Model lists and serialisation
# -----------------------------
#
# In a typical analysis scenario a model consists of multiple model
# components, or a “catalog” or “source library”. To handle this list of
# multiple model components, Gammapy has a `Models` class:
#

from gammapy.modeling.models import Models

models = Models([model, diffuse])
print(models)


######################################################################
# Individual model components in the list can be accessed by their name:
#

print(models["my-source"])


######################################################################
# **Note:** To make the access by name unambiguous, models are required to
# have a unique name, otherwise an error will be thrown.
#
# To see which models are available you can use the ``.names`` attribute:
#

print(models.names)


######################################################################
# Note that a `SkyModel` object can be evaluated for a given longitude,
# latitude, and energy, but the `Models` object cannot. This `Models`
# container object will be assigned to `Dataset` or `Datasets`
# together with the data to be fitted. Checkout e.g. the
# :doc:`/tutorials/api/model_management` tutorial for details.
#
# The `~gammapy.modeling.models.Models` class also has in place ``.append()`` and ``.extend()``
# methods:
#

model_copy = model.copy(name="my-source-copy")
models.append(model_copy)


######################################################################
# This list of models can be also serialised to a custom YAML based
# format:
#

models_yaml = models.to_yaml()
print(models_yaml)


######################################################################
# The structure of the yaml files follows the structure of the python
# objects. The ``components`` listed correspond to the `SkyModel` and
# components of the ``Models``. For each ``SkyModel``
# we have information about its ``name``, ``type`` (corresponding to the
# tag attribute) and sub-mobels (i.e ``spectral`` model and eventually
# ``spatial`` model). Then the spatial and spectral models are defined by
# their type and parameters. The ``parameters`` keys name/value/unit are
# mandatory, while the keys min/max/frozen are optionnals (so you can
# prepare shorter files).
#
# If you want to write this list of models to disk and read it back later
# you can use:
#

models.write("models.yaml", overwrite=True)

models_read = Models.read("models.yaml")


######################################################################
# Additionally the models can exported and imported togeter with the data
# using the ``Datasets.read()`` and ``Datasets.write()`` methods as shown
# in the :doc:`/tutorials/analysis-3d/analysis_mwl`
# notebook.
#
# Models with shared parameter
# ----------------------------
#
# A model parameter can be shared with other models, for example we can
# define two power-law models with the same spectral index but different
# amplitudes:
#

pwl2 = PowerLawSpectralModel()
pwl2.index = pwl.index
pwl.index.value = (
    2.3  # also update pwl2 as the parameter object is now the same as shown below
)
print(pwl.index)
print(pwl2.index)


######################################################################
# In the YAML files the shared parameter is flagged by the additional
# ``link`` entry that follows the convention ``parameter.name@unique_id``:
#

models = Models([SkyModel(pwl, name="source1"), SkyModel(pwl2, name="source2")])
models_yaml = models.to_yaml()
print(models_yaml)


######################################################################
# .. _custom-model:
#
# Implementing a custom model
# ---------------------------
#
# In order to add a user defined spectral model you have to create a
# SpectralModel subclass. This new model class should include:
#
# -  a tag used for serialization (it can be the same as the class name)
# -  an instantiation of each Parameter with their unit, default values
#    and frozen status
# -  the evaluate function where the mathematical expression for the model
#    is defined.
#
# As an example we will use a PowerLawSpectralModel plus a Gaussian (with
# fixed width). First we define the new custom model class that we name
# ``MyCustomSpectralModel``:
#

from gammapy.modeling import Parameter
from gammapy.modeling.models import SpectralModel


class MyCustomSpectralModel(SpectralModel):
    """My custom spectral model, parametrising a power law plus a Gaussian spectral line.

    Parameters
    ----------
    amplitude : `astropy.units.Quantity`
        Amplitude of the spectra model.
    index : `astropy.units.Quantity`
        Spectral index of the model.
    reference : `astropy.units.Quantity`
        Reference energy of the power law.
    mean : `astropy.units.Quantity`
        Mean value of the Gaussian.
    width : `astropy.units.Quantity`
        Sigma width of the Gaussian line.

    """

    tag = "MyCustomSpectralModel"
    amplitude = Parameter("amplitude", "1e-12 cm-2 s-1 TeV-1", min=0, is_norm=True)
    index = Parameter("index", 2, min=0)
    reference = Parameter("reference", "1 TeV", frozen=True)
    mean = Parameter("mean", "1 TeV", min=0)
    width = Parameter("width", "0.1 TeV", min=0, frozen=True)

    @staticmethod
    def evaluate(energy, index, amplitude, reference, mean, width):
        pwl = PowerLawSpectralModel.evaluate(
            energy=energy,
            index=index,
            amplitude=amplitude,
            reference=reference,
        )
        gauss = amplitude * np.exp(-((energy - mean) ** 2) / (2 * width**2))
        return pwl + gauss


######################################################################
# It is good practice to also implement a docstring for the model,
# defining the parameters and also definig a ``.tag``, which specifies the
# name of the model for serialisation. Also note that gammapy assumes that
# all SpectralModel evaluate functions return a flux in unit of
# `"cm-2 s-1 TeV-1"` (or equivalent dimensions).
#
# This model can now be used as any other spectral model in Gammapy:
#

my_custom_model = MyCustomSpectralModel(mean="3 TeV")
print(my_custom_model)

print(my_custom_model.integral(1 * u.TeV, 10 * u.TeV))

plt.figure()
my_custom_model.plot(energy_bounds=[1, 10] * u.TeV)


######################################################################
# As a next step we can also register the custom model in the
# ``SPECTRAL_MODELS`` registry, so that it becomes available for
# serilisation:
#

SPECTRAL_MODEL_REGISTRY.append(MyCustomSpectralModel)

model = SkyModel(spectral_model=my_custom_model, name="my-source")
models = Models([model])
models.write("my-custom-models.yaml", overwrite=True)

# !cat my-custom-models.yaml


######################################################################
# Similarly you can also create custom spatial models and add them to the
# ``SPATIAL_MODELS`` registry. In that case gammapy assumes that the
# evaluate function return a normalized quantity in “sr-1” such as the
# model integral over the whole sky is one.
#


######################################################################
# Models with energy dependent morphology
# ---------------------------------------
#
# A common science case in the study of extended sources is to probe for
# energy dependent morphology, eg: in Supernova Remnants or Pulsar Wind
# Nebulae. Traditionally, this has been done by splitting the data into
# energy bands and doing individual fits of the morphology in these energy
# bands.
#
# `~gammapy.modeling.models.SkyModel` offers a natural framework to simultaneously model the
# energy and morphology, e.g. spatial extent described by a parametric
# model expression with energy dependent parameters.
#
# The models shipped within gammapy use a “factorised” representation of
# the source model, where the spatial (:math:`l,b`), energy (:math:`E`)
# and time (:math:`t`) dependence are independent model components and not
# correlated:
#
# :raw-latex:`\begin{align}f(l, b, E, t) = F(l, b) \cdot G(E) \cdot H(t)\end{align}`
#
# To use full 3D models, ie $f(l, b, E) = F(l, b, E)
# :raw-latex:`\cdot`\ G(E) $, you have to implement your own custom
# `SpatialModel`. Note that it is still necessary to multiply by a
# `SpectralModel`, :math:`G(E)` to be dimensionally consistent.
#
# In this example, we create Gaussian Spatial Model with the extension
# varying with energy. For simplicity, we assume a linear dependence on
# energy and parameterize this by specifying the extension at 2 energies.
# You can add more complex dependences, probably motivated by physical
# models.
#

from astropy.coordinates.angle_utilities import angular_separation
from gammapy.modeling.models import SpatialModel


class MyCustomGaussianModel(SpatialModel):
    """My custom Energy Dependent Gaussian model.

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
    is_energy_dependent = True
    lon_0 = Parameter("lon_0", "0 deg")
    lat_0 = Parameter("lat_0", "0 deg", min=-90, max=90)

    sigma_1TeV = Parameter("sigma_1TeV", "2.0 deg", min=0)
    sigma_10TeV = Parameter("sigma_10TeV", "0.2 deg", min=0)

    @staticmethod
    def evaluate(lon, lat, energy, lon_0, lat_0, sigma_1TeV, sigma_10TeV):

        sep = angular_separation(lon, lat, lon_0, lat_0)

        # Compute sigma for the given energy using linear interpolation in log energy
        sigma_nodes = u.Quantity([sigma_1TeV, sigma_10TeV])
        energy_nodes = [1, 10] * u.TeV
        log_s = np.log(sigma_nodes.to("deg").value)
        log_en = np.log(energy_nodes.to("TeV").value)
        log_e = np.log(energy.to("TeV").value)
        sigma = np.exp(np.interp(log_e, log_en, log_s)) * u.deg

        exponent = -0.5 * (sep / sigma) ** 2
        norm = 1 / (2 * np.pi * sigma**2)
        return norm * np.exp(exponent)


######################################################################
# Serialisation of this model can be achieved as explained in the previous
# section. You can now use it as standard ``SpatialModel`` in your
# analysis. Note that this is still a ``SpatialModel`` and not a
# ``SkyModel``, so it needs to be multiplied by a ``SpectralModel`` as
# before.
#

spatial_model = MyCustomGaussianModel()
spectral_model = PowerLawSpectralModel()
sky_model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)

print(spatial_model.evaluation_radius)


######################################################################
# To visualise it, we evaluate it on a 3D geom.
#

energy_axis = MapAxis.from_energy_bounds(
    energy_min=0.1 * u.TeV, energy_max=10.0 * u.TeV, nbin=3, name="energy_true"
)
geom = WcsGeom.create(skydir=(0, 0), width=5.0 * u.deg, binsz=0.1, axes=[energy_axis])

plt.figure()
spatial_model.plot_grid(geom=geom, add_cbar=True, figsize=(14, 3))


######################################################################
# For computational purposes, it is useful to specify a
# ``evaluation_radius`` for ``SpatialModels`` - this gives a size on which
# to compute the model. Though optional, it is highly recommended for
# Custom Spatial Models. This can be done, for ex, by defining the
# following function inside the above class:
#


@property
def evaluation_radius(self):
    """Evaluation radius (`~astropy.coordinates.Angle`)."""
    return 5 * np.max([self.sigma_1TeV.value, self.sigma_10TeV.value]) * u.deg
