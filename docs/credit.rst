.. include:: references.txt

.. _credit:

Credit
======

This is a short summary of who contributed what to gammapy.

* Christoph Deil (MPIK Heidelberg) started gammapy in 2013 and is maintaining it.
* Contributions by Axel Donath (MPIK Heidelberg):
    - Some code committed by Christoph Deil (e.g. functionality in `gammapy.image`)
      are actually code written by Axel for his bachelor and master's thesis. 
    - Various image analysis and simulation functionality.
* Contributions by Ellis Owen (MPIK Heidelberg):
    - Various bugfixes.
* Contributions by Regis Terrier (APC Paris): 
    - Continuous wavelet transform for images with Poisson statistics (`GH PR 25 <https://github.com/gammapy/gammapy/pull/25>`__)
* Martin Raue (DESY, Hamburg) wrote `PyFACT`_ in 2012 to prototype FITS gamma-ray
  data analysis as part of the first CTA data challenge.
  Martin has left astronomy and the last commit for PyFACT was in June 2012.
  In February 2014 Christoph Deil integrated the PyFACT functionality in gammapy
  (see `GH PR 68 <https://github.com/gammapy/gammapy/pull/68>`__
  for details which parts of gammapy originate from PyFACT).

For further information see the
`GitHub gammapy contributors page <https://github.com/gammapy/gammapy/graphs/contributors>`__
and the `Oloh gammapy summary page <https://www.ohloh.net/p/gammapy>`__.

Note that because gammapy started out as a fork of the
`Astropy affiliated package template <https://github.com/astropy/package-template>`__,
the automatic contributors summary on Github and Ohloh is not accurate in the sense
that it contains the contributions to the (non-gammapy-specific) astropy affiliated project template. 

.. _PyFACT: http://pyfact.readthedocs.org


