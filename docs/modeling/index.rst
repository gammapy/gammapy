.. include:: ../references.txt

.. _modeling:

*****************************
modeling - Models and fitting
*****************************

.. currentmodule:: gammapy.modeling

Introduction
============

`gammapy.modeling` contains all the functionality related to modeling and fitting
data. This includes spectral, spatial and temporal model classes, as well as the fit
and parameter API. A list of available model can be found further down this page.
In general the models are grouped into the following categories:

- `~gammapy.modeling.models.SpectralModel`: models to describe spectral shapes of sources
- `~gammapy.modeling.models.SpatialModel`: models to describe spatial shapes (morphologies) of sources
- `~gammapy.modeling.models.TemporalModel`: models to describe temporal flux evolution of sources, such as light and phase curves
- `~gammapy.modeling.models.SkyModel` and `~gammapy.modeling.models.SkyDiffuseCube`: model to combine the spectral and spatial model components

The models follow a naming scheme which contains the category as a suffix to the class
name.

Getting Started
===============

Spectral Models
---------------
Models are imported from the `gammapy.modeling.models` namespace:

.. code:: python

	from gammapy.modeling.models import PowerLawSpectralModel

	pwl = PowerLawSpectralModel()
	print(pwl)

Spectral models all come with default parameters. Different parameter
values can be passed on creation of the model, either as a string defining
the value and unit or as an `~astropy.units.Quantity` object directly:

.. code:: python

	from astropy import units as u
	from gammapy.modeling.models import PowerLawSpectralModel

	pwl = PowerLawSpectralModel(amplitude="2.7e-12 TeV-1 cm-2 s-1", index=2.2)
	print(pwl)

	amplitude = 1e-12 * u.Unit("TeV-1 cm-2 s-1")
	pwl = PowerLawSpectralModel(amplitude=amplitude, index=2.2)
	print(pwl)

The model can be evaluated at given energies by calling the model instance:

.. code:: python

	from astropy import units as u
	from gammapy.modeling.models import PowerLawSpectralModel

	pwl = PowerLawSpectralModel()

	energy = [1, 3, 10, 30] * u.TeV
	dnde = pwl(energy)
	print(dnde)


For spectral models you can computed in addition the integrated and energy flux
in a given energy range:

.. code:: python

	from astropy import units as u
	from gammapy.modeling.models import PowerLawSpectralModel

	pwl = PowerLawSpectralModel(index=2.2)

	flux = pwl.integral(emin=1 * u.TeV, emax=10 * u.TeV)
	print(flux)

	eflux = pwl.energy_flux(emin=1 * u.TeV, emax=10 * u.TeV)
	print(eflux)

	# with an array of emins and emaxs
	energy = [1, 3, 10, 30] * u.TeV
	flux = pwl.integral(emin=energy[:-1], emax=energy[1:])

Spatial Models
--------------

Spatial models are imported from the same `gammapy.modeling.models` namespace:

.. code:: python

	from gammapy.modeling.models import GaussianSpatialModel

	gauss = GaussianSpatialModel(lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg")
	print(gauss)

The default coordinate frame is ``"icrs"``, but the frame can be modified using the
``frame`` argument:

.. code:: python

	from gammapy.modeling.models import GaussianSpatialModel

	gauss = GaussianSpatialModel(lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg", frame="galactic")
	print(gauss)

	print(gauss.position)

Spatial models can be evaluated again by calling the instance:

.. code:: python

	from gammapy.modeling.models import GaussianSpatialModel
	from astropy import units as u

	gauss = GaussianSpatialModel(lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg")

	lon = [0, 0.1] * u.deg
	lat = [0, 0.1] * u.deg

	flux_per_omega = gauss(lon, lat)

The returned quantity corresponds to a surface brightness. Spatial model
can be also evaluated using `gammapy.maps.Map` and `gammapy.maps.Geom` objects:

.. code:: python

	import matplotlib.pyplot as plt
	from gammapy.modeling.models import GaussianSpatialModel
	from gammapy.maps import Map

	gauss = GaussianSpatialModel(lon_0="0 deg", lat_0="0 deg", sigma="0.2 deg")
	m = Map.create(skydir=(0, 0), width=(1, 1), binsz=0.02)
	m.quantity = gauss.evaluate_geom(m.geom)
	m.plot()
	plt.show()


SkyModel and SkyDiffuseCube
---------------------------

The `~gammapy.modeling.models.SkyModel` class combines a spectral and a spatial model. It can be created
from existing spatial and spectral model components:

.. code:: python

	from gammapy.modeling.models import SkyModel, GaussianSpatialModel, PowerLawSpectralModel

	pwl = PowerLawSpectralModel()
	gauss = GaussianSpatialModel("0 deg", "0 deg", "0.2 deg")

	source = SkyModel(spectral_model=pwl, spatial_model=gauss, name="my-source")
	print(source)

In addition a ``name`` can be assigned to model as an identifier for models with
multiple components.

The `gammapy.modeling.models.SkyDiffuseCube` can be used to represent source models based on templates.
It can be created from an existing FITS file:

.. code:: python

	from gammapy.modeling.models import SkyDiffuseCube

	diffuse = SkyDiffuseCube.read("$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz")
	print(diffuse)


Fitting
-------

For examples how to fit models to data please check out the following tutorials:

- :gp-notebook:`analysis_3d`
- :gp-notebook:`spectrum_analysis`


Reference/API
=============

.. automodapi:: gammapy.modeling
    :no-inheritance-diagram:
    :include-all-objects:

.. automodapi:: gammapy.modeling.models
    :no-inheritance-diagram:
    :include-all-objects:
