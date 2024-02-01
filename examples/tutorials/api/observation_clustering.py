"""
Observational clustering
========================

Clustering observations into specific groups.


Context
-------

Typically, observations from gamma-ray telescopes can span a number of
different observation periods, therefore it is likely that the observation
conditions and quality are not always the same. This tutorial aims to provide
a way in which observations can be grouped such that similar observations are grouped
together, and then the data reduction is performed.


Objective
---------

To cluster similar observations based on various quantities.

Proposed approach
-----------------

For completeness two different methods for grouping of observations are shown here.

- A simple grouping based on zenith angle from an existing observations table.

- Grouping the observations depending on the IRF quality, by means of hierarchical clustering.

"""


import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from gammapy.data import DataStore
from gammapy.data.observations import Observations
from gammapy.data.utils import get_irfs_features
from gammapy.utils.cluster import hierarchical_clustering, standard_scaler

######################################################################
# Obtain the observations
# -----------------------
#
# First need to define the `~gammapy.data.DataStore` object for the HESS DL3 DR1
# data. Next, utilise a cone search to select only the observations of interest.
# In this case, we choose PKS 2155-304 as the object of interest.
#
# The `~gammapy.data.ObservationTable` is then filtered using the `select_observations` tool.
#

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")

selection = dict(
    type="sky_circle",
    frame="icrs",
    lon="329.71693826 deg",
    lat="-30.2255 deg",
    radius="2 deg",
)
obs_table = data_store.obs_table
selected_obs_table = obs_table.select_observations(selection)


######################################################################
# More complex selection can be done by utilising the obs_table entries directly.
# We can now retrieve the relevant observations by passing their obs_id to the
# `~gammapy.data.DataStore.get_observations` method.
#

obs_ids = selected_obs_table["OBS_ID"]
observations = data_store.get_observations(obs_ids)


######################################################################
# Show various observation quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Print here the range of zenith angles and muon efficiencies, to see
# if there is a sensible way to group the observations.
#

obs_zenith = selected_obs_table["ZEN_PNT"].to(u.deg)
obs_muoneff = selected_obs_table["MUONEFF"]

print(f"{np.min(obs_zenith):.2f} < zenith angle < {np.max(obs_zenith):.2f}")
print(f"{np.min(obs_muoneff):.2f} < muon efficiency < {np.max(obs_muoneff):.2f}")


######################################################################
# Manual grouping of observations
# -------------------------------
#
# Here we can plot the zenith angle vs muon efficiency of the observations.
# We decide to group the observations according to their zenith angle.
# This is done manually as per a user defined cut, in this case we take the
# median value of the zenith angles to define each observation group.
#
# This type of grouping can be utilised according to different parameters i.e.
# zenith angle, muon efficiency, offset angle. The quantity chosen can therefore
# be adjusted according to each specific science case.
#

median_zenith = np.median(obs_zenith)

labels = []
for obs in observations:
    zenith = obs.get_pointing_altaz(time=obs.tmid).zen
    labels.append("low_zenith" if zenith < median_zenith else "high_zenith")
grouped_observations = observations.group_by_label(labels)

print(grouped_observations)

######################################################################
#
# The results for each group of observations is shown visually below.
#

fix, ax = plt.subplots(1, 1, figsize=(7, 5))
for obs in grouped_observations["group_low_zenith"]:
    ax.plot(
        obs.get_pointing_altaz(time=obs.tmid).zen,
        obs.meta.optional["MUONEFF"],
        "d",
        color="red",
    )
for obs in grouped_observations["group_high_zenith"]:
    ax.plot(
        obs.get_pointing_altaz(time=obs.tmid).zen,
        obs.meta.optional["MUONEFF"],
        "o",
        color="blue",
    )
ax.set_ylabel("Muon efficiency")
ax.set_xlabel("Zenith angle (deg)")
ax.axvline(median_zenith.value, ls="--", color="black")
plt.show()


######################################################################
# This shows the observations grouped by zenith angle. The diamonds
# are observations which have a zenith angle less than the median value,
# whilst the circles are observations above the median.
#
# The `grouped_observations` provide a list of `~gammapy.data.Observations`
# which can be utilised in the usual way to show the various properties
# of the observations i.e. see the :doc:`/tutorials/data/cta` tutorial.
#


######################################################################
# Hierarchical clustering of observations
# ---------------------------------------
#
# This method shows how to cluster observations based on their IRF quantities,
# in this case those that have a similar edisp and psf. The
# `~gammapy.data.utils.get_irfs_features` is utilised to achieve this. The
# observations are then clustered based on these criteria using
# `~gammapy.utils.cluster.hierarchical_clustering`. The idea here is to minimise
# the variance of both edisp and psf within a specific group to limit the error
# on the quantity when they are stacked at the dataset level.
#
# In this example, the irf features are computed for the `edisp-res` and
# `psf-radius` at 1 TeV. This is stored as a `~astropy.table.table.Table`, as shown below.
#

source_position = SkyCoord(329.71693826 * u.deg, -30.2255890 * u.deg, frame="icrs")
names = ["edisp-res", "psf-radius"]
features_irfs = get_irfs_features(
    observations, energy_true="1 TeV", position=source_position, names=names
)
print(features_irfs)

######################################################################
# Compute standardized features by removing the mean and scaling to unit
# variance:
#

scaled_features_irfs = standard_scaler(features_irfs)
print(scaled_features_irfs)


######################################################################
# The `~gammapy.utils.cluster.hierarchical_clustering` then clusters
# this table into ``t=2`` groups with a corresponding label for each group.
# In this case, we choose to cluster the observations into two groups.
# We can print this table to show the corresponding label which has been
# added to the previous ``feature_irfs`` table.
#
# The arguments for `~scipy.cluster.hierarchy.fcluster` are passed as
# a dictionary here.
#

features = hierarchical_clustering(scaled_features_irfs, fcluster_kwargs={"t": 2})
print(features)

######################################################################
# Finally, ``observations.group_by_label`` creates a dictionary containing ``t``
# `~gammapy.data.Observations` objects by grouping the similar labels.
#

obs_clusters = observations.group_by_label(features["labels"])
print(obs_clusters)


mask_1 = features["labels"] == 1
mask_2 = features["labels"] == 2
fix, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.set_ylabel("edisp-res")
ax.set_xlabel("psf-radius")
ax.plot(
    features_irfs[mask_1]["edisp-res"],
    features_irfs[mask_1]["psf-radius"],
    "d",
    color="green",
    label="Group 1",
)
ax.plot(
    features_irfs[mask_2]["edisp-res"],
    features_irfs[mask_2]["psf-radius"],
    "o",
    color="magenta",
    label="Group 2",
)
ax.legend()
plt.show()


######################################################################
# The groups here are divided by the quality of the IRFs values ``edisp-res``
# and ``psf-radius``. The diamond and circular points indicate how the observations
# are grouped.
#
#
# In both examples we have a set of `~gammapy.data.Observation` objects which
# can be reduced using the `~gammapy.makers.DatasetsMaker` to create two (in this
# specific case) separate datasets. These can then be jointly fitted using the
# :doc:`/tutorials/analysis-3d/analysis_mwl` tutorial.
#
