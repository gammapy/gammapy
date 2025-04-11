"""
Make a theta-square plot
========================

This tutorial explains how to make such plot, that is the distribution
of event counts as a function of the squared angular distance to a test
position.

"""


######################################################################
# Setup
# -----
#

# %matplotlib inline
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from gammapy.data import DataStore
from gammapy.maps import MapAxis
from gammapy.makers.utils import make_theta_squared_table
from gammapy.visualization import plot_theta_squared_table


######################################################################
# Check setup
# -----------
#

from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


######################################################################
# Get some data
# -------------
#
# Here, some data taken on the Crab by the H.E.S.S. test data released are
# used.
#

data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
observations = data_store.get_observations([23523, 23526])


######################################################################
# Defined a test position
# -----------------------
#

position = SkyCoord.from_name("crab")
print(position)


######################################################################
# Creation of the theta2 plot
# ---------------------------
#

theta2_axis = MapAxis.from_bounds(0, 0.2, nbin=20, interp="lin", unit="deg2")
theta2_table = make_theta_squared_table(
    observations=observations,
    position=position,
    theta_squared_axis=theta2_axis,
)

plt.figure(figsize=(10, 5))
plot_theta_squared_table(theta2_table)
plt.show()
