.. _dataformats_target_lists:

Target lists
============

There are some Gammapy tools that run analyses in batch mode,
and to simplify this we define a simple CSV "target list" format here.

A typical application is that there are many (potential) target positions to analyse,
but the "target list" could also contain the same target position several times,
varying other parameters.

CSV format
----------

We use the `CSV <https://en.wikipedia.org/wiki/Comma-separated_values>`_ (comma-separated values) format for target lists.

A target list must have at least a column called ``Number`` with unique entries:

.. code-block:: text

   Number
   42
   43

Usually it has many more columns with information about each target:

.. code-block:: text

   Number,Name,RA,DEC
   42,Crab,83.633212,22.014460

Special column names that the Gammapy batch processing tools understand:

* ``Number`` --- Target number
* ``Name`` --- Target name
* ``RA``, ``DEC`` or ``GLON``, ``GLAT`` -- Target position in Equatorial (ICRS) or Galactic coordinates (deg)
* ``Theta`` --- Source region radius (deg)
* ``FOV`` --- Field of view radius (deg)

Usually there will be other, tool-specific parameter columns.
