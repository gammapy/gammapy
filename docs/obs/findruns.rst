.. _obs_findruns:

findruns
========

TODO: reference run list format

The ``gammapy-findruns`` command line tool can be used to select a subset of runs from a given run list.

Examples:

* Find all runs within 5 deg of the Galactic center:

  .. code-block:: bash

      $ gammapy-findruns cone --x 0 --y 0 --r 5 --system galactic \
                         --in all_runs.lis --out galactic_center_runs.lis

* Select all runs in a box along the Galactic plane (GLON = -20 .. +20 deg, GLAT = -3 .. +3 deg):

  .. code-block:: bash

      $ gammapy-findruns box --x 0 --y 0 --dx 20 --dy 3 --system galactic \
                         --in all_runs.lis  --out galactic_plane_runs.lis

* Select all runs in a given date range (can of course be combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-findruns --date_min 2010-04-26 --date_max "2010-04-29 12:42" \
                         --in all_runs.lis --out run_042_to_100.lis

* Select all runs in a given run range (can of course be combined with other selections shown above):

  .. code-block:: bash

      $ gammapy-findruns --run_min 42 --run_max 100 \
                         --in all_runs.lis --out run_042_to_100.lis

Using ``gammapy-findruns`` is easy, you don't have to remember all the options, just type:

  .. code-block:: bash

      $ findruns --help

at the command line or read the usage help (TODO: add link here).

.. note:: Noel Dawe's ``goodruns`` tool for `ATLAS <http://atlas.ch>`__ run selection
      (`docs <http://ndawe.github.io/goodruns/>`__, `code <https://github.com/ndawe/goodruns>`__)
      is a nice example for a run selection tool.
