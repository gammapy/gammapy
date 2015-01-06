.. _tutorials-gammapy-pfmap:

Skymaps with ``gammapy-pfmap`` and Sherpa
=========================================

The script ``gammapy-pfmap`` is used to create binned skymaps from FITS event lists.
It calculates background maps using the ring and the template background method
(if a corresponding template background eventlist is provided)
and produces signal, background, excess, and significance maps.
These maps can be written to fits files and then viewed and analyzed with standard fits tools, e.g., fv, ds9, or sherpa.

Running ``gammapy-pfmap``
-------------------------

Using ``gammapy-pfmap`` is straight forward.
To create skymaps from a file ``data.fits`` using the object position from the header of the file
as center of skymap and writing the skymaps to FITS files (option: ``-w``), use:

.. code-block:: bash

   $ gammapy-pfmap data.fits -w

``gammapy-pfmap`` can also handle compressed files (gzip).

If you want to analyse several files together,
you need to create an ASCII file containing the filenames
(first string per row is used; bankfile), e.g.,:

.. code-block:: bash

   $ ls *.fits.gz > my.bnk
   $ python scripts/pfmap.py my.bnk -w

You can change the parameters of the skymap via command line options, e.g.,:

.. code-block:: bash

   $ gammapy-pfmap my.bnk -w -s 4. -b 0.1 -r 0.1 -c "(83.633, 22.0145)"

creating a skymap of size 4 deg (``-s``) with a bin size 0.1 deg (``-c``)
and correlation radius for the oversampled skymap of 0.1 deg (``-r``).
The center of the map is set to RA=83.633 deg, Dec=22.0145 deg (J2000; -).
Check the ``--help`` option for more details on the different command line options.

After running ``gammapy-pfmap`` with the option ``-w`` you will find a couple of new FITS files
in you working directory starting with ``skymap_ring`` (or ``skymap_templ``).
Files containing the string overs contain correlated/oversampled maps.

The other string identifier are as follows:

* ac = Acceptance
* al = Alpha factor
* bg = Background
* ev = Events
* ex = Excess
* si = Significance

You can view the files with  with standard FITS tools, e.g., fv or ds9.

Morphology fitting with Sherpa
------------------------------

Find below an example python script, which shows to fit an excess skymap with a 2D double gaussian function with Sherpa.
For this to work it is assumed that you have the python packages sherpa, pyfits, and kapteyn installed on your machine.

.. literalinclude:: sherpa_fit_image.py
    :linenos:
    :language: python
