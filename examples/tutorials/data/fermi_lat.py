"""
Fermi-LAT with Gammapy
======================

Data inspection and preliminary analysis with Fermi-LAT data.

Introduction
------------

This tutorial will show you how to work with Fermi-LAT data with
Gammapy. As an example, we will look at the Galactic center region using
the high-energy dataset that was used for the 3FHL catalog, in the
energy range 10 GeV to 2 TeV.

We note that support for Fermi-LAT data analysis in Gammapy is very
limited. For most tasks, we recommend you use
`Fermipy <http://fermipy.readthedocs.io/>`__, which is based on the
`Fermi Science
Tools <https://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`__
(Fermi ST).

Using Gammapy with Fermi-LAT data could be an option for you if you want
to do an analysis that is not easily possible with Fermipy and the Fermi
Science Tools. For example a joint likelihood fit of Fermi-LAT data with
data e.g. from H.E.S.S., MAGIC, VERITAS or some other instrument, or
analysis of Fermi-LAT data with a complex spatial or spectral model that
is not available in Fermipy or the Fermi ST.

Besides Gammapy, you might want to look at are
`Sherpa <http://cxc.harvard.edu/sherpa/>`__ or
`3ML <https://threeml.readthedocs.io/>`__. Or just using Python to roll
your own analysis using several existing analysis packages. E.g. it it
possible to use Fermipy and the Fermi ST to evaluate the likelihood on
Fermi-LAT data, and Gammapy to evaluate it e.g. for IACT data, and to do
a joint likelihood fit using
e.g. `iminuit <http://iminuit.readthedocs.io/>`__ or
`emcee <http://dfm.io/emcee>`__.

To use Fermi-LAT data with Gammapy, you first have to use the Fermi ST
to prepare an event list (using `gtselect` and `gtmktime`, exposure
cube (using `gtexpcube2` and PSF (using `gtpsf`). You can then use
`~gammapy.data.EventList`, `~gammapy.maps` and the
`~gammapy.irf.PSFMap` to read the Fermi-LAT maps and PSF, i.e. support
for these high level analysis products from the Fermi ST is built in. To
do a 3D map analysis, you can use Fit for Fermi-LAT data in the same way
that it’s use for IACT data. This is illustrated in this notebook. A 1D
region-based spectral analysis is also possible, this will be
illustrated in a future tutorial.

Setup
-----

**IMPORTANT**: For this notebook you have to get the prepared `3fhl`
dataset provided in your `$GAMMAPY_DATA`.

Note that the `3fhl` dataset is high-energy only, ranging from 10 GeV
to 2 TeV.

"""

from astropy import units as u
from astropy.coordinates import SkyCoord

# %matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display
from gammapy.data import EventList
from gammapy.datasets import Datasets, MapDataset
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    Models,
    PointSpatialModel,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()

######################################################################
# Events
# ------
#
# To load up the Fermi-LAT event list, use the `~gammapy.data.EventList`
# class:
#

events = EventList.read("$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_events_selected.fits.gz")
print(events)


######################################################################
# The event data is stored in a
# `astropy.table.Table <http://docs.astropy.org/en/stable/api/astropy.table.Table.html>`__
# object. In case of the Fermi-LAT event list this contains all the
# additional information on position, zenith angle, earth azimuth angle,
# event class, event type etc.
#

print(events.table.colnames)

display(events.table[:5][["ENERGY", "RA", "DEC"]])

print(events.time[0].iso)
print(events.time[-1].iso)

energy = events.energy
energy.info("stats")


######################################################################
# As a short analysis example we will count the number of events above a
# certain minimum energy:
#

for e_min in [10, 100, 1000] * u.GeV:
    n = (events.energy > e_min).sum()
    print(f"Events above {e_min:4.0f}: {n:5.0f}")


######################################################################
# Counts
# ------
#
# Let us start to prepare things for an 3D map analysis of the Galactic
# center region with Gammapy. The first thing we do is to define the map
# geometry. We chose a TAN projection centered on position
# `(glon, glat) = (0, 0)` with pixel size 0.1 deg, and four energy bins.
#

gc_pos = SkyCoord(0, 0, unit="deg", frame="galactic")
energy_axis = MapAxis.from_edges(
    [1e4, 3e4, 1e5, 3e5, 2e6], name="energy", unit="MeV", interp="log"
)
counts = Map.create(
    skydir=gc_pos,
    npix=(100, 80),
    proj="TAN",
    frame="galactic",
    binsz=0.1,
    axes=[energy_axis],
    dtype=float,
)
# We put this call into the same Jupyter cell as the Map.create
# because otherwise we could accidentally fill the counts
# multiple times when executing the `fill_by_coord` multiple times.
counts.fill_events(events)

print(counts.geom.axes[0])

counts.sum_over_axes().smooth(2).plot(stretch="sqrt", vmax=30)
plt.show()

######################################################################
# Exposure
# --------
#
# The Fermi-LAT dataset contains the energy-dependent exposure for the
# whole sky as a HEALPix map computed with `gtexpcube2`. This format is
# supported by `~gammapy.maps.Map` directly.
#
# Interpolating the exposure cube from the Fermi ST to get an exposure
# cube matching the spatial geometry and energy axis defined above with
# Gammapy is easy. The only point to watch out for is how exactly you want
# the energy axis and binning handled.
#
# Below we just use the default behaviour, which is linear interpolation
# in energy on the original exposure cube. Probably log interpolation
# would be better, but it doesn’t matter much here, because the energy
# binning is fine. Finally, we just copy the counts map geometry, which
# contains an energy axis with ``node_type="edges"``. This is non-ideal
# for exposure cubes, but again, acceptable because exposure doesn’t vary
# much from bin to bin, so the exact way interpolation occurs in later use
# of that exposure cube doesn’t matter a lot. Of course you could define
# any energy axis for your exposure cube that you like.
#

exposure_hpx = Map.read("$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_exposure_cube_hpx.fits.gz")
print(exposure_hpx.geom)
print(exposure_hpx.geom.axes[0])

exposure_hpx.plot()
plt.show()

######################################################################
# For exposure, we choose a geometry with ``node_type='center'``,
axis = MapAxis.from_energy_bounds(
    "10 GeV",
    "2 TeV",
    nbin=10,
    per_decade=True,
    name="energy_true",
)
geom = WcsGeom(wcs=counts.geom.wcs, npix=counts.geom.npix, axes=[axis])

exposure = exposure_hpx.interp_to_geom(geom)

print(exposure.geom)
print(exposure.geom.axes[0])

######################################################################
# Exposure is almost constant across the field of view
exposure.slice_by_idx({"energy_true": 0}).plot(add_cbar=True)
plt.show()

######################################################################
# Exposure varies very little with energy at these high energies
energy = [10, 100, 1000] * u.GeV
print(exposure.get_by_coord({"skycoord": gc_pos, "energy_true": energy}))


######################################################################
# Galactic diffuse background
# ---------------------------
#


######################################################################
# The Fermi-LAT collaboration provides a galactic diffuse emission model,
# that can be used as a background model for Fermi-LAT source analysis.
#
# Diffuse model maps are very large (100s of MB), so as an example here,
# we just load one that represents a small cutout for the Galactic center
# region.
#
# In this case, the maps are already in differential units, so we do not
# want to normalise it again.
#

template_diffuse = TemplateSpatialModel.read(
    filename="$GAMMAPY_DATA/fermi-3fhl-gc/gll_iem_v06_gc.fits.gz", normalize=False
)

print(template_diffuse.map)

diffuse_iem = SkyModel(
    spectral_model=PowerLawNormSpectralModel(),
    spatial_model=template_diffuse,
    name="diffuse-iem",
)


######################################################################
# Let’s look at the map of first energy band of the cube:
#
template_diffuse.map.slice_by_idx({"energy_true": 0}).plot(add_cbar=True)
plt.show()


######################################################################
# Here is the spectrum at the Galactic center:
#

dnde = template_diffuse.map.to_region_nd_map(region=gc_pos)
dnde.plot()
plt.xlabel("Energy (GeV)")
plt.ylabel("Flux (cm-2 s-1 MeV-1 sr-1)")
plt.show()


######################################################################
# Isotropic diffuse background
# ----------------------------
#
# To load the isotropic diffuse model with Gammapy, use the
# `~gammapy.modeling.models.TemplateSpectralModel`. We are using
# `'extrapolate': True` to extrapolate the model above 500 GeV:
#

filename = "$GAMMAPY_DATA/fermi_3fhl/iso_P8R2_SOURCE_V6_v06.txt"

diffuse_iso = create_fermi_isotropic_diffuse_model(
    filename=filename, interp_kwargs={"extrapolate": True}
)


######################################################################
# We can plot the model in the energy range between 50 GeV and 2000 GeV:
#
energy_bounds = [50, 2000] * u.GeV
diffuse_iso.spectral_model.plot(energy_bounds, yunits=u.Unit("1 / (cm2 MeV s)"))
plt.show()


######################################################################
# PSF
# ---
#
# Next we will tke a look at the PSF. It was computed using `gtpsf`, in
# this case for the Galactic center position. Note that generally for
# Fermi-LAT, the PSF only varies little within a given regions of the sky,
# especially at high energies like what we have here. We use the
# `~gammapy.irf.PSFMap` class to load the PSF and use some of it’s
# methods to get some information about it.
#

psf = PSFMap.read("$GAMMAPY_DATA/fermi_3fhl/fermi_3fhl_psf_gc.fits.gz", format="gtpsf")
print(psf)


######################################################################
# To get an idea of the size of the PSF we check how the containment radii
# of the Fermi-LAT PSF vary with energy and different containment
# fractions:
#

plt.figure(figsize=(8, 5))
psf.plot_containment_radius_vs_energy()
plt.show()


######################################################################
# In addition we can check how the actual shape of the PSF varies with
# energy and compare it against the mean PSF between 50 GeV and 2000 GeV:
#

plt.figure(figsize=(8, 5))

energy = [100, 300, 1000] * u.GeV
psf.plot_psf_vs_rad(energy_true=energy)

spectrum = PowerLawSpectralModel(index=2.3)
psf_mean = psf.to_image(spectral_model=spectrum)
psf_mean.plot_psf_vs_rad(c="k", ls="--", energy_true=[500] * u.GeV)

plt.xlim(1e-3, 0.3)
plt.ylim(1e3, 1e6)
plt.legend()
plt.show()

######################################################################
# This is what the corresponding PSF kernel looks like:
#

psf_kernel = psf.get_psf_kernel(
    position=geom.center_skydir, geom=geom, max_radius="1 deg"
)
psf_kernel.to_image().psf_kernel_map.plot(stretch="log", add_cbar=True)
plt.show()


######################################################################
# Energy Dispersion
# ~~~~~~~~~~~~~~~~~
#
# For simplicity we assume a diagonal energy dispersion:
#

e_true = exposure.geom.axes["energy_true"]
edisp = EDispKernelMap.from_diagonal_response(
    energy_axis_true=e_true, energy_axis=energy_axis
)

edisp.get_edisp_kernel().plot_matrix()
plt.show()


######################################################################
# Fit
# ---
#
# Now, the big finale: let’s do a 3D map fit for the source at the
# Galactic center, to measure it’s position and spectrum. We keep the
# background normalization free.
#

spatial_model = PointSpatialModel(lon_0="0 deg", lat_0="0 deg", frame="galactic")
spectral_model = PowerLawSpectralModel(
    index=2.7, amplitude="5.8e-10 cm-2 s-1 TeV-1", reference="100 GeV"
)

source = SkyModel(
    spectral_model=spectral_model,
    spatial_model=spatial_model,
    name="source-gc",
)

models = Models([source, diffuse_iem, diffuse_iso])

dataset = MapDataset(
    models=models,
    counts=counts,
    exposure=exposure,
    psf=psf,
    edisp=edisp,
    name="fermi-dataset",
)

# %%time
fit = Fit()
result = fit.run(datasets=[dataset])

print(result)

print(models)

residual = counts - dataset.npred()

residual.sum_over_axes().smooth("0.1 deg").plot(
    cmap="coolwarm", vmin=-3, vmax=3, add_cbar=True
)
plt.show()

######################################################################
# Serialisation
# -------------
#
# To serialise the created dataset, you must proceed through the
# Datasets API
#

Datasets([dataset]).write(
    filename="fermi_dataset.yaml", filename_models="fermi_models.yaml", overwrite=True
)
datasets_read = Datasets.read(
    filename="fermi_dataset.yaml", filename_models="fermi_models.yaml"
)
print(datasets_read)


######################################################################
# Exercises
# ---------
#
# -  Fit the position and spectrum of the source `SNR
#    G0.9+0.1 <http://gamma-sky.net/#/cat/tev/110>`__.
# -  Make maps and fit the position and spectrum of the `Crab
#    nebula <http://gamma-sky.net/#/cat/tev/25>`__.
#


######################################################################
# Summary
# -------
#
# In this tutorial you have seen how to work with Fermi-LAT data with
# Gammapy. You have to use the Fermi ST to prepare the exposure cube and
# PSF, and then you can use Gammapy for any event or map analysis using
# the same methods that are used to analyse IACT data.
#
# This works very well at high energies (here above 10 GeV), where the
# exposure and PSF is almost constant spatially and only varies a little
# with energy. It is not expected to give good results for low-energy
# data, where the Fermi-LAT PSF is very large. If you are interested to
# help us validate down to what energy Fermi-LAT analysis with Gammapy
# works well (e.g. by re-computing results from 3FHL or other published
# analysis results), or to extend the Gammapy capabilities (e.g. to work
# with energy-dependent multi-resolution maps and PSF), that would be very
# welcome!
#
