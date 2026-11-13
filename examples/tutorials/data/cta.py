"""
CTAO with Gammapy
=================

Access and inspect CTAO data and instrument response functions (IRFs) using Gammapy.

Introduction
------------

The `Cherenkov Telescope Array Observatory (CTAO) <https://www.ctao.org/>`__ is the next generation
ground-based observatory for gamma-ray astronomy. Gammapy is the core
library for the Cherenkov Telescope Array Observatory (CTAO) science tools
(`CTAO Press
Release <https://www.ctao.org/news/ctao-adopts-the-gammapy-software-package-for-science-analysis/>`__).

CTAO will start taking data in the coming years. A full simulation of 7 years of observations has been
produced and is openly accessible to the community. You can learn how to analyse CTAO data and to use Gammapy
using these data.

If you are a member of the CTAO consortium, you can also use older simulated dataset from previous internal
CTAO data challenges which ran in 2018 (see `here <https://forge.in2p3.fr/projects/data-challenge-1-dc-1/wiki>`__)
and  2024 (see `here <https://ctaoobservatory.sharepoint.com/:f:/r/sites/share-open-data/Shared%20Documents/Reference%20datasets/Internal%20Science%20Data%20Challenge?csf=1&web=1&e=gNuFzI>`__)

You can also use Gammapy to simulate CTAO data and evaluate CTAO performance
using the CTAO response files. The latest ones, for the Beta array configuration (prod 6, 2026), are available
on Zenodo: https://zenodo.org/records/22871179.

Additionally, two sets of responses are available for older array layouts: the `Omega (prod3b, 2016) <https://zenodo.org/records/5163273>`__,
and `Alpha (prod5, 2021) <https://zenodo.org/records/5499840>`__ array configurations.

They are all fully supported by Gammapy.


Tutorial overview
-----------------

This notebook shows how to access CTAO data and instrument response
functions (IRFs) using Gammapy, and gives some examples how to quick
look the content of CTAO files, especially to see the shape of CTAO IRFs.

At the end of the notebooks, we give several links to other tutorial
notebooks that show how to simulate CTAO data and how to evaluate CTAO
observability and sensitivity, or how to analyse CTAO data.

Note that the FITS data and IRF format currently used by CTAO is the one
documented at https://gamma-astro-data-formats.readthedocs.io/, and is
also used by H.E.S.S. and other Imaging Atmospheric Cherenkov Telescopes
(IACTs). So if you see other Gammapy tutorials using e.g. H.E.S.S.
example data, know that they also apply to CTAO, all you have to do is to
change the loaded data or IRFs to CTAO.

Setup
-----

"""

import os
from pathlib import Path
from astropy import units as u

# %matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display
from gammapy.data import DataStore
from gammapy.irf import load_irf_dict_from_file


######################################################################
# CTA SDC
# -------
#
# The CTAO open data challenge
#
# **For convenience**, since the SDC data files are large, we have taken
# a tiny subset of the data, and **included them at** `$GAMMAPY_DATA/cta-sdc`.
# Data are distributed for different sites (North or South) and night sky
# conditions (Dark or Moon).
#
# Files
# ~~~~~
#
# Next we will show a quick overview of the files and how to load them,
# and some quick look plots showing the shape of the CTAO IRFs. How to do
# CTAO simulations and analyses is shown in other tutorials, see links at
# the end of this notebook.
#


######################################################################
# Datastore
# ~~~~~~~~~
#
# Gammapy provides the `~gammapy.data.DataStore` interface to access the data.
# It can be created via the index files when they are provided.
# Index files consist in two tables:
# - the HDU index table which connects every observation element to its location on disk
# - the observation table which provides a summary of observation metadata
#

data_store_south = DataStore.from_dir("$GAMMAPY_DATA/cta-sdc/South/Dark")
print(data_store_south)

######################################################################
# The tables can be directly explored
#

display(data_store_south.obs_table)

######################################################################
# In the absence of index files, it is possible to generate the
# `~gammapy.data.DataStore` directly from the SDC event files.
#

path = Path(os.environ["GAMMAPY_DATA"]) / "cta-sdc/North/Dark/"
paths = list(path.rglob("dl3-event-obs*.fits.gz"))
data_store_north = DataStore.from_events_files(paths)
print(data_store_north)

display(data_store_north.obs_table)

######################################################################
# Accessing an observation can then be done directly using its OBS_ID.
# Here we consider an observation from the Southern CTAO array.
# A `~gammapy.data.Observation` bundles the event list and the IRFs as
# well as Good Time Intervals (GTI) and metadata.
#
# SDC IRFs are full-containment response functions which are split in:
# -  effective area
# -  energy dispersion
# -  point spread function
# -  background
#

observation = data_store_south.obs(5000000013)
print(observation)
observation.peek()


######################################################################
# Events
# ~~~~~~
#
# Events data are directly accessible via the `events` attribute of
# the `~gammapy.data.Observation`.
#
#
# Events can be accessed from the observation object like:

events = observation.events

######################################################################
# Here we print the data from the first 5 events listed in the table:
#

display(events.table[:5])

######################################################################
# And show a summary plot:
#

events.peek()
plt.show()


######################################################################
# We can access the observation metadata:
#

# array used
print(observation.meta.obs_info.instrument)
# target name
print(observation.meta.target.name)


######################################################################
# Effective area
# ~~~~~~~~~~~~~~
#

aeff = observation.aeff
print(aeff)

aeff.peek()
plt.show()

######################################################################
# Energy dispersion
# ~~~~~~~~~~~~~~~~~
#

observation.edisp.peek()
plt.show()


######################################################################
# Point spread function
# ~~~~~~~~~~~~~~~~~~~~~
#

observation.psf.peek()
plt.show()

######################################################################
# Background
# ~~~~~~~~~~
#
# The background is given as a rate in units `MeV-1 s-1 sr-1`.
#

observation.bkg.peek()
plt.show()


######################################################################
# Working with CTAO IRFs
# ----------------------
#
# To study expected performances of a given CTAO observation, it can be useful
# to build a synthetic `~gammapy.data.Observation`. For this, you need to retrieve
# the CTAO instrument response functions. They are distributed for the
# north and south arrays, for different observing conditions, i.e.
# different zenith angles and night sky brightnesses.
#
# The following IRFs are available:
#
# -  effective area
# -  energy dispersion
# -  point spread function
# -  background
#
# The Beta array configuration (prod 6, 2026) IRFs are available
# on Zenodo: https://zenodo.org/records/22871179.
#
# We ship a small subset of these IRFs in `$GAMMAPY_DATA/cta-caldb`.
# Gammapy provides an helper function to read in those files and return
# a dictionary of `~gammapy.irf.IRF` objects.
#

irf_filename = "$GAMMAPY_DATA/cta-caldb/Prod6-CTAO-North-20deg-AverageAz-4LSTs09MSTs-dark-180000s-v1.0.fits.gz"
irfs = load_irf_dict_from_file(irf_filename)
print(irfs)

######################################################################
# To visualise the background at particular energies:
#

irfs["bkg"].plot_at_energy(["50 GeV", "100 GeV", "500 GeV", "1 TeV", "3 TeV", "10 TeV"])
plt.show()

######################################################################
# Point spread function
# ~~~~~~~~~~~~~~~~~~~~~
#

irfs["psf"].peek()
plt.show()


# %%
# This is how for analysis you could slice out the PSF
# at a given field of view offset
plt.figure(figsize=(8, 5))
irfs["psf"].plot_containment_radius_vs_energy(
    offset=[1] * u.deg, fraction=[0.68, 0.8, 0.95]
)
plt.show()


######################################################################
# Latest CTAO IRFs
# ~~~~~~~~~~~~~~~~
#
# If you want to use other response files, the following code cells (remove the # to uncomment)
# explain how to proceed. This example is made with the Beta configuration (Prod6).
#

# import os
# !curl -o ctao-prod6-zenodo-v1.0-fits-only.zip https://zenodo.org/records/22871179/files/ctao-prod6-zenodo-v1.0-fits-only.zip
# !unzip ctao-prod6-zenodo-v1.0-fits-only.zip
# os.chdir("ctao-prod6-zenodo-v1.0-fits-only")
# !ls fits/

# !tar xf fits/CTAO-Performance-Prod6-CTAO-North-52deg-halfmoon-v1.0.FITS.tar.gz -C fits/.
# !ls fits/*.fits.gz

# irfs1 = load_irf_dict_from_file("fits/Prod6-CTAO-North-52deg-NorthAz-4LSTs09MSTs-halfmoon-180000s-v1.0.fits.gz")
# irfs1["aeff"].plot_energy_dependence()

# irfs2 = load_irf_dict_from_file("fits/Prod6-CTAO-North-52deg-SouthAz-4LSTs09MSTs-halfmoon-180000s-v1.0.fits.gz")
# irfs2["aeff"].plot_energy_dependence()


######################################################################
# Next steps
# ----------
#
# -  Learn how to analyse data with
#    :doc:`/tutorials/starting/analysis_1` and
#    :doc:`/tutorials/starting/analysis_2` or any other
#    Gammapy analysis tutorial.
# -  Learn how to evaluate CTAO observability and sensitivity with
#    :doc:`/tutorials/analysis-3d/simulate_3d`,
#    :doc:`/tutorials/analysis-1d/spectrum_simulation`
#    or :doc:`/tutorials/analysis-1d/cta_sensitivity`.
#
