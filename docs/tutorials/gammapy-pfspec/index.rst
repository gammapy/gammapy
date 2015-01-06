.. _tutorials-gammapy-pfspec:

Spectra with ``gammapy-pfspec`` and Sherpa
==========================================

Running ``gammapy-pfspec``
--------------------------

The script ``gammapy-pfspec`` produces PHA spectrum files from FITS event lists, which can be analyzed with tools like XSPEC.
The instrument response is taken from ARF (effective area) and RMF (energy distribution matrix) files
and is assumed to be constant over the duration of a data segment (run).
The background is estimated using a ring at the same camera/FoV distance as the source, cutting out the source position.

Per data file, ``gammapy-pfspec`` needs three inputs:
the name of the data file and the corresponding ARF and RMF file names.
These can be given via command line but usually it is more efficient to create an ASCII file (bankfile),
with each row giving the data file name, the ARF and the RMF file names, separate by spaces.
We assume, such a bankfile has been created for the data called my.bnk.

To create the pha files run:

.. code-block:: bash

   $ gammapy-pfspec my.bnk -w -r 0.125

The option ``-r`` denotes the radius in degree of the circular source region from which the spectrum will be extracted (theta cut).
This should match the cut used in the ARF files.

This will produce three PHA files per data file in the working directory:

* bg = Background
* excess = Excess
* signal = Signal (i.e. excess = signal - background)

Spectrum fitting with Sherpa
----------------------------

The output PHA files can be analyzed with spectra fitting tools like XSPEC or Sherpa.

Find below an example session for XSPEC.

Note that XSPEC and Sherpa do not recognize the units given in the ARF/RMF files correctly,
always assuming keV and cm^2.
Therefore, the fit results have to be converted correspondingly.
Some output has been omitted below and has been replaced with dots (...).

.. literalinclude:: xspec_session.txt
    :language: text
