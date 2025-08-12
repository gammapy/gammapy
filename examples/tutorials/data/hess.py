"""
H.E.S.S. with Gammapy
=====================

Explore H.E.S.S. event lists and IRFs.


Introduction
------------

`H.E.S.S. <https://www.mpi-hd.mpg.de/hfm/HESS/>`__ is an array of
gamma-ray telescopes located in Namibia. Gammapy is regularly used and
fully supports H.E.S.S. high level data analysis, after export to the
current `open data level 3
format <https://gamma-astro-data-formats.readthedocs.io/>`__.

The H.E.S.S. data is private, and H.E.S.S. analysis is mostly documented
and discussed in the internal Wiki pages and in
H.E.S.S.-internal communication channels. However, in 2018, a small
sub-set of archival H.E.S.S. data was publicly released, called the
`H.E.S.S. DL3
DR1 <https://www.mpi-hd.mpg.de/hfm/HESS/pages/dl3-dr1/>`__, the data
level 3, data release number 1. This dataset is 50 MB in size and is
used in many Gammapy analysis tutorials, and can be downloaded via
`gammapy
download <https://docs.gammapy.org/dev/getting-started/index.html#quickstart-setup>`__.

This notebook is a quick introduction to this specific DR1 release. It
briefly describes H.E.S.S. data and instrument responses and show a
simple exploration of the data with the creation of theta-squared plot.

H.E.S.S. members can find details on the DL3 FITS production on this
`Confluence
page <https://cchesswiki.in2p3.fr/en/hess/working_groups/analysis_and_reconstruction_working_group/ar_active_tasks/hess_fits_data>`__
and access more detailed tutorials in the BitBucket `hess-open-tools` repository.

DL3 DR1
-------

This is how to access data and IRFs from the H.E.S.S. data level 3, data
release 1.

"""

import astropy.units as u

# %matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display
from gammapy.data import DataStore


######################################################################
# A useful way to organize the relevant files are the index tables. The
# observation index table contains information on each particular run,
# such as the pointing, or the run ID. The HDU index table has a row per
# relevant file (i.e., events, effective area, psf…) and contains the path
# to said file. Together they can be loaded into a Datastore by indicating
# the directory in which they can be found, in this case
# `$GAMMAPY_DATA/hess-dl3-dr1`:
#

######################################################################
# Create and get info on the data store

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")

data_store.info()

######################################################################
# Preview an excerpt from the observation table

display(data_store.obs_table[:2][["OBS_ID", "DATE-OBS", "RA_PNT", "DEC_PNT", "OBJECT"]])

######################################################################
# Get a single observation

obs = data_store.obs(23523)

######################################################################
# Select and peek events

obs.events.select_offset([0, 2.5] * u.deg).peek()

######################################################################
# Peek the effective area

obs.aeff.peek()

######################################################################
# Peek the energy dispersion

obs.edisp.peek()

######################################################################
# Peek the psf
obs.psf.peek()

######################################################################
# Peek the background rate
obs.bkg.to_2d().plot()
plt.show()


######################################################################
# Exercises
# ---------
#
# -  Find the `OBS_ID` for the runs of the Crab nebula
# -  Compute the expected number of background events in the whole RoI for
#    `OBS_ID=23523` in the 1 TeV to 3 TeV energy band, from the
#    background IRF.
#


######################################################################
# Next steps
# ----------
#
# Now you know how to access and work with H.E.S.S. data. All other
# tutorials and documentation apply to H.E.S.S. and CTAO or any other IACT
# that provides DL3 data and IRFs in the standard format.
#
# You can see the following tutorials for more detailed analysis using H.E.S.S. data
#
# -  :doc:`/tutorials/starting/analysis_1`
# -  :doc:`/tutorials/starting/analysis_2`
# -  :doc:`/tutorials/analysis-1d/spectral_analysis`
# -  :doc:`/tutorials/analysis-1d/extended_source_spectral_analysis`
# -  :doc:`/tutorials/analysis-2d/ring_background`
#
