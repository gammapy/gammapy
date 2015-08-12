.. _background_make_models:

Make models
===========

Gammapy tools to produce background models.

Make cube models
----------------

The ``gammapy-make_bg_cube_models`` command line tool can be used to produce
background cube models from the data files at a given path location.

For more details, please refer to `~gammapy.scripts.make_bg_cube_models`.

Examples
~~~~~~~~

The ``gammapy-make_bg_cube_models`` tool has a few options. For a full list of options, please use:

  .. code-block:: bash

      $ gammapy-make_bg_cube_models --help

at the command line.

Command examples:

* Create background models (can take a few minutes):

  .. code-block:: bash

      $ gammapy-make_bg_cube_models /path/to/fits/event_lists/base/dir

* Run a quick test using only a few runs:

  .. code-block:: bash

      $ gammapy-make_bg_cube_models /path/to/fits/event_lists/base/dir \
                         --test True

The output files are created in the current directory:

* ``bg_observation_table.fits.gz``: total observatin table used for
  the models. The list has been filtered from observations taken at
  or nearby known sources.

* ``splitted_obs_list/``: directory containing the runlist splitted
  according to the selected binning for the background runs.

* ``bg_cube_models/``: directory containing the background models in
  2 different formats: *tables*, for data analysis and *images* for
  a quick visualization using eg. ``DS9``.
