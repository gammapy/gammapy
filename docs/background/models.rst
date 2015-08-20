.. _bg_models:

Background Models
=================

The naming of the models in this section follows the convention from
:ref:`dataformats_overview`.

The documentation on how to produce background models in Gammapy is
available at :ref:`background_make_models`.

.. _background_3D:

BACKGROUND_3D
-------------

**BACKGROUND_3D** is a background rate 3D cube *(X, Y, energy)* in
units of per energy, per time, per solid angle. **X** and **Y** are
given in detector coordinates **(DETX, DETY)**, a.k.a.
**nominal system**. This is a tangential system to the instrument
during observations.

The `~gammapy.background.CubeBackgroundModel` is used as container class for this model.
It has methods to read, write and operate the 3D cubes.

For the moment, only I/O and visualization methods are implemented.
A test file is located in the `~gammapy-extra` repository
(`bg_cube_model_test.fits <https://github.com/gammapy/gammapy-extra/blob/master/test_datasets/background/bg_cube_model_test.fits>`_).
The file has been produced with `~gammapy.datasets.make_test_bg_cube_model`.

An example script of how to read/write the files and perform some
simple plots is given in the `examples` directory:
:download:`plot_bg_cube_model.py <../../examples/plot_bg_cube_model.py>`

.. literalinclude:: ../../examples/plot_bg_cube_model.py

The data of the cube can be accessed via:

.. code:: python

   energy_bin = bg_cube_model.find_energy_bin(energy=Quantity(2., 'TeV'))
   det_bin = bg_cube_model.find_det_bin(det=Angle([0., 0.], 'degree'))
   bg_cube_model.background[energy_bin, det_bin[1], det_bin[0]]

More complex plots can be easily produced with a few lines of code:

.. plot:: background/plot_bgcube.py
   :include-source:

There is also a method in the `~gammapy.datasets` module called
`~gammapy.datasets.make_test_bg_cube_model` for creating test
`~gammapy.background.CubeBackgroundModel` objects.

In order to compare 2 sets of background cube models, the following
script in the `examples` directory can be used:
:download:`plot_bg_cube_model_comparison.py
<../../examples/plot_bg_cube_model_comparison.py>`
