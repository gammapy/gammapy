.. _obs_observation_grouping:

Observation grouping
====================

Observation grouping can be very helpful to speed the analysis of
data with similar characteristics. It is also essential in some cases
, like the production of background models. In this case, the scarce
statistic forces the grouping of observations taken under similar
conditions in order to produce reliable models.

There are 2 classes in Gammapy that handle observation groups:

* `~gammapy.data.ObservationGroupAxis` is a class to define an axis
  along which to define bins for creating observation groups. The
  class offers support for dimensionless or
  `~astropy.units.Quantity`-like parameter axes. In both cases, both,
  continuous and discrete variables can be used.

* `~gammapy.data.ObservationGroups` is a class that takes a list of
  axes (`~gammapy.data.ObservationGroupAxis`) and defines groups
  based on the cartesian product of the bins on each axis. The group
  definitions are internally stored as a `~astropy.table.Table`
  object. This class has also methods to group the observations of a
  given 'ObservationTable' following the defined grouping.

Examples
--------
Create a few axes and define the corresponding observation grouping:

.. code:: python

    alt = Angle([0, 30, 60, 90], 'degree')
    az = Angle([-90, 90, 270], 'degree')
    ntels = np.array([3, 4])
    list_obs_group_axis = [ObservationGroupAxis('ALT', alt, 'bin_edges'),
                           ObservationGroupAxis('AZ', az, 'bin_edges'),
                           ObservationGroupAxis('N_TELS', ntels, 'bin_values')]
    obs_groups = ObservationGroups(list_obs_group_axis)

Print the observation group table (group definitions):

>>> print(obs_groups.obs_groups_table)

The output looks like this:

.. code::

    GROUP_ID ALT_MIN ALT_MAX AZ_MIN AZ_MAX N_TELS
               deg     deg    deg    deg         
    -------- ------- ------- ------ ------ ------
           0     0.0    30.0  -90.0   90.0      3
           1     0.0    30.0  -90.0   90.0      4
           2     0.0    30.0   90.0  270.0      3
           3     0.0    30.0   90.0  270.0      4
           4    30.0    60.0  -90.0   90.0      3
           5    30.0    60.0  -90.0   90.0      4
           6    30.0    60.0   90.0  270.0      3
           7    30.0    60.0   90.0  270.0      4
           8    60.0    90.0  -90.0   90.0      3
           9    60.0    90.0  -90.0   90.0      4
          10    60.0    90.0   90.0  270.0      3
          11    60.0    90.0   90.0  270.0      4

Print the observation group axes:

>>> print(obs_groups.info)

The output looks like this:

.. code::

    ALT bin_edges [  0.  30.  60.  90.] deg
    AZ bin_edges [ -90.   90.  270.] deg
    N_TELS bin_values [3 4]

Group the observations of an observation list and print them:

>>> obs_table_grouped = obs_groups.group_observation_table(obs_table)
>>> print(obs_table_grouped)

This would print an observation table with the format described in
:ref:`gadf:obs-index` and an extra-column at the
beginning specifying the ID of the group to which each observation
belongs.

Get the observations of a particular group and print them:

>>> obs_table_group8 = obs_groups.get_group_of_observations(obs_table_grouped, 8)
>>> print(obs_table_group8)

This would print the observation table corresponding to the group
with ID 8.
