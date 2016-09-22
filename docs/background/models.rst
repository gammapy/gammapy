.. _bg_models:

Background models
=================

The naming of the models in this section follows the convention from
:ref:`dataformats_overview`.

The documentation on how to produce background models in Gammapy is
available at :ref:`background_make_background_models`.

.. _background_3D:

BACKGROUND_3D
-------------

**BACKGROUND_3D** is a background rate 3D cube *(X, Y, energy)* in
units of per energy, per time, per solid angle. **X** and **Y** are
given in detector coordinates **(DETX, DETY)**, a.k.a.
**nominal system**. This is a tangential system to the instrument
during observations.

Two classes are used as container for this model:

* The `~gammapy.background.FOVCube` class is used as base container for
  cubes. It has generic methods to I/O (read/write) and operate the
  3D cubes. It also has visualization methods to plot slices/bins of
  the cubes.

* The `~gammapy.background.FOVCubeBackgroundModel` class is used to
  contain and handle cube background models.
  It contains 3 cubes of type `~gammapy.background.FOVCube`:

    * ``counts_cube`` - counts (a.k.a. events) used to fill the model.
    * ``livetime_cube``- livetime correction used for the model.
    * ``background_cube`` - background model (rate)

  The class also defines usefull methods to produce the models, such
  as define binning, fill (histogram) the model or smooth.

Two test files are located in the ``gammapy-extra`` repository as
examples and test benches of these classes:

* `bg_cube_model_test1.fits`_ is a `~gammapy.background.FOVCube` produced with an older version of
  `~gammapy.datasets.make_test_bg_cube_model`, using a simplified
  background model. The current version of the mehod produces a
  `~gammapy.background.FOVCubeBackgroundModel` object.

* `bg_cube_model_test2.fits.gz`_ is a `~gammapy.background.FOVCubeBackgroundModel` produced with
  `~gammapy.background.make_bg_cube_model`, using dummy data produced
  with `~gammapy.datasets.make_test_dataset`.

An example script of how to read/write the cubes from file and
perform some simple plots is given in the ``examples`` directory:
:download:`example_plot_background_model.py <../../examples/example_plot_background_model.py>`

.. literalinclude:: ../../examples/example_plot_background_model.py

The data of the cube can be accessed via:

.. code:: python

   energy_bin = bg_cube_model.find_energy_bin(energy=Quantity(2., 'TeV'))
   det_bin = bg_cube_model.find_det_bin(det=Angle([0., 0.], 'degree'))
   bg_cube_model.background[energy_bin, det_bin[1], det_bin[0]]

More complex plots can be easily produced with a few lines of code:

.. plot:: background/plot_bgcube.py
   :include-source:


.. _bg_cube_model_test1.fits: https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/background/bg_cube_model_test1.fits
.. _bg_cube_model_test2.fits.gz: https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/background/bg_cube_model_test2.fits.gz
