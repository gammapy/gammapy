.. _obs_select:

Observation and data selection
==============================

The ``gammapy-data-select`` command line tool can be used to select a
subset of observations from a given observation list. The format for
observation lists is described in :ref:`dataformats_observation_lists`.

This tool works with FITS files as input/output for now.

For more details, please refer to `~gammapy.scripts.obs_select`.

Examples
--------

The ``gammapy-data-select`` tool has many options. Only a few examples
are shown here. For a full list of options, please use:

.. code-block:: bash

    $ gammapy-data-select --help

at the command line.

In order to test the examples below, the test observation list
file located in the ``gammapy-extra`` repository
(`test_observation_table.fits <https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/obs/test_observation_table.fits>`_)
can be used as input observation list.

* Find all observations within 5 deg of the Galactic center:

  .. code-block:: bash

      $ gammapy-data-select all_obs.fits galactic_center_obs.fits \
                         --x 0 --y 0 --r 50 --system 'galactic'

* Select all observations in a box along the Galactic plane
  (GLON = -20 .. +20 deg, GLAT = -3 .. +3 deg) (can of course be
  combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-data-select all_obs.fits galactic_plane_obs.fits \
                         --x 0 --y 0 --dx 20 --dy 3 --system 'galactic'

* Select all observations in a given date range (can of course be
  combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-data-select all_obs.fits obs_2010-04-26_to_2010-04-29-12h42.fits \
                         --t_start '2012-04-20' --t_stop '2012-04-30T12:42'

* Select all observations in a given observation ID range (can of
  course be combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-data-select all_obs.fits obs_042_to_100.fits \
                         --par_name 'OBS_ID' --par_min 42 --par_max 101

* Select all observations in a given altitude range (can of course
  be combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-data-select all_obs.fits alt_70_to_90_deg_obs.fits \
                         --par_name 'ALT' --par_min 70 --par_max 90

* Select all observations with exactly 4 telescopes (can of course
  be combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-data-select all_obs.fits 4_tel_obs.fits \
                         --par_name 'N_TELS' --par_min 4 --par_max 4
