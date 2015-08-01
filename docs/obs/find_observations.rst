.. _obs_find_observations:

Find observations
=================

TODO: reference observation list/table format

The ``gammapy-find-obs`` command line tool can be used to select a
subset of observations from a given observation list.

It works with `fits` files as input/output.

Examples:

* Find all observations within 5 deg of the Galactic center:

  .. code-block:: bash

      $ gammapy-find-obs all_obs.fits galactic_center_obs.fits \
                         --x 0 --y 0 --r 50 --system 'galactic'

* Select all observations in a box along the Galactic plane
  (GLON = -20 .. +20 deg, GLAT = -3 .. +3 deg) (can of course be
  combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-find-obs all_obs.fits galactic_plane_obs.fits \
                         --x 0 --y 0 --dx 20 --dy 3 --system 'galactic'

* Select all observations in a given date range (can of course be
  combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-find-obs all_obs.fits obs_2010-04-26_to_2010-04-29-12h42.fits \
                         --t_start '2010-04-26' --t_stop '2010-04-29T12:42'

* Select all observations in a given observation ID range (can of
  course be combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-find-obs all_obs.fits obs_042_to_100.fits \
                         --par_name 'OBS_ID' --par_min 41 --par_max 101

* Select all observations in a given altitude range (can of course
  be combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-find-obs all_obs.fits alt_70_to_90_deg_obs.fits \
                         --par_name 'ALT' --par_min 70 --par_max 90

Using ``gammapy-find-obs`` is easy, you don't have to remember all
the options, just type:

  .. code-block:: bash

      $ gammapy-find-obs --help

at the command line or read the usage help (TODO: add link here).

.. note:: Noel Dawe's ``goodruns`` tool for `ATLAS <http://atlas.ch>`__ run selection
      (`docs <http://ndawe.github.io/goodruns/>`__, `code <https://github.com/ndawe/goodruns>`__)
      is a nice example for a run selection tool.
