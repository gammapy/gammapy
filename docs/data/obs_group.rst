.. _obs_observation_grouping:

Observation grouping
====================

Observation grouping can be very helpful to speed the analysis of
data with similar characteristics. It is also essential in some cases,
like the production of background models. In this case, the scarce
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

Create an `~gammapy.data.ObservationGroups` object with three axes:

.. code-block:: python

    from astropy.coordinates import Angle
    from gammapy.data import ObservationGroups, ObservationGroupAxis
    zenith = Angle([0, 30, 40, 50], 'deg')
    ntels = [3, 4]
    obs_groups = ObservationGroups([
        ObservationGroupAxis('ZENITH', zenith, fmt='edges'),
        ObservationGroupAxis('N_TELS', ntels, fmt='values'),
    ])


The axes info is stored:

.. code-block:: python

    >>> print(obs_groups.info)
    ZENITH edges [  0.  30.  40.  50.] deg
    N_TELS values [3 4]

The observation groups are stored as a table (computed once on ``ObservationGroups`` object construction):

.. code-block:: python

    >>> print(obs_groups.obs_groups_table)
    GROUP_ID ZENITH_MIN ZENITH_MAX N_TELS
                deg        deg
    -------- ---------- ---------- ------
           0        0.0       30.0      3
           1        0.0       30.0      4
           2       30.0       40.0      3
           3       30.0       40.0      4
           4       40.0       50.0      3
           5       40.0       50.0      4

TODO: make this a real example ... use the four Crab runs!

Apply the observation grouping to an observation list::

    >>> obs_table = obs_groups.apply(obs_table)
    >>> print(obs_table)

This would print an observation table with the format described in
:ref:`gadf:obs-index` and an extra-column at the beginning specifying the ID
of the group to which each observation belongs.

Get the observations of a particular group and print them::

    >>> obs_table_group2 = obs_groups.get_group_of_observations(obs_table, 2)
    >>> print(obs_table_group2)

This would print the observation table corresponding to the group
with ID 8.
