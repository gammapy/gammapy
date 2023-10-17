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
together, and then the data reduction in performed.


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
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from gammapy.data import DataStore
from gammapy.data.utils import get_irfs_features
from gammapy.utils.cluster import hierarchical_clustering

######################################################################
# Obtain the observations
# -----------------------
#
# Create the data store and obtain the observations from the H.E.S.S.
# DL3-DR1 for PKS 2155-304.

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
obs_id = data_store.obs_table["OBS_ID"][
    data_store.obs_table["OBJECT"] == "PKS 2155-304"
]
observations = data_store.get_observations(obs_id)


######################################################################
# Show various observation quantities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Print here the range of zenith angles and muon efficiencies, to see
# if there is a sensible way to group the observations.

obs_zenith = []
obs_muoneff = []

for obs in observations:
    obs_zenith.append(obs.get_pointing_altaz(time=obs.tmid).zen.deg)
    obs_muoneff.append(obs.obs_info["MUONEFF"])

print(f"{np.min(obs_zenith):.2f} deg < zenith angle < {np.max(obs_zenith):.2f} deg")
print(f"{np.min(obs_muoneff):.2f} < muon efficiency < {np.max(obs_muoneff):.2f}")


######################################################################
# Split the observations by zenith angle
# --------------------------------------
#
# Here we can plot the zenith angle vs muon efficiency of the
# observations. We decide to group the observations according to their
# zenith angle. The # median value of the zenith angles is used to
# define these groups. These are shown visually below.
#

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)

obs_A = []
obs_B = []
for obs in observations:
    zenith = obs.get_pointing_altaz(time=obs.tmid).zen.deg
    if zenith < np.median(obs_zenith):
        ax.plot(zenith, obs.obs_info["MUONEFF"], "o", color="red", alpha=0.4)
        obs_A.append(obs.obs_id)
    if zenith > np.median(obs_zenith):
        ax.plot(zenith, obs.obs_info["MUONEFF"], "o", color="blue", alpha=0.4)
        obs_B.append(obs.obs_id)

ax.set_ylabel("Muon efficiency")
ax.set_xlabel("Zenith angle (deg)")
ax.axvline(28.9, ls="--", color="black")


######################################################################
# This shows the observation group by zenith angle. The red points
# are observations which have a zenith angle less than the median value,
# whilst the blue points are observations above the median.
#
# These groups can then be used to create two separate datasets which
# can be analysed utilising a joint fit method.


######################################################################
# Group observations according to IRF quantities
# ----------------------------------------------
#
# This method shows how to cluster observations that have a similar edisp
# and psf from the IRFs. The `gammapy.data.utils.get_irfs_features` is
# utilised to achieve this. The observations are then clustered based on
# this criteria using `gammapy.utils.cluster.hierarchical_clustering`. The
# idea here is to minimise the variance of both edisp and psf within a specific
# group to limit the error on the quantity when they are stacked at the dataset level.
#

source_position = SkyCoord.from_name("PKS 2155-304")
names = ["edisp-bias", "edisp-res", "psf-radius"]
features_irfs = get_irfs_features(
    observations, energy_true="1 TeV", position=source_position, names=names
)
features = hierarchical_clustering(
    features_irfs, fcluster_kwargs={"t": 2}
)  # Allow only two groups for the cluster
obs_clusters = observations.group_by_label(features["labels"])


fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111)
ax.set_ylabel("Muon efficiency")
ax.set_xlabel("Zenith angle (deg)")

for obs in obs_clusters["group_1"]:
    ax.plot(
        obs.get_pointing_altaz(time=obs.tmid).zen.deg,
        obs.obs_info["MUONEFF"],
        "o",
        color="green",
        alpha=0.4,
    )
for obs in obs_clusters["group_2"]:
    ax.plot(
        obs.get_pointing_altaz(time=obs.tmid).zen.deg,
        obs.obs_info["MUONEFF"],
        "o",
        color="magenta",
        alpha=0.4,
    )


######################################################################
# The groups here are divided by the quality of the IRFs values edisp
# and psf. The green and magenta points indicate how the observations
# are grouped.
