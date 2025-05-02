"""
Make a theta-square plot
========================

This tutorial explains how to make such a plot, that is the distribution
of event counts as a function of the squared angular distance, to a test
position.

"""


######################################################################
# Setup
# -----
#

# %matplotlib inline
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
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


######################################################################
# Making a theta2 plot for a given energy range
# ---------------------------------------------
#
# with the function `make_theta_squared_table`, one can also select a fixed energy range.
#

theta2_table_en = make_theta_squared_table(
    observations=observations,
    position=position,
    theta_squared_axis=theta2_axis,
    energy_edges=[1.2, 11] * u.TeV,
)
plt.figure(figsize=(10, 5))
plot_theta_squared_table(theta2_table_en)
plt.show()


######################################################################
# Statistical significance of a detection
# ---------------------------------------
#
# To get the significance of a signal, the usual method consists of using the reflected background method
# (see the maker tutorial: :doc:`/user-guide/makers/reflected`) to compute the WStat statistics
# (see :ref:`wstat`, :doc:`/user-guide/stats/fit_statistics`). This is the well-known method of Li&Ma [LiMa1983]_
# using ON and OFF regions.
# The following tutorials show how to get an excess significance:
#  * :doc:`/tutorials/analysis-1d/spectral_analysis`
#  * :doc:`/tutorials/analysis-1d/extended_source_spectral_analysis`
#
