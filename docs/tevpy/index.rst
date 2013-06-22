`tevpy`
=======

What is it?
-----------

`tevpy` is an open source (BSD licensed) TeV gamma-ray astronomy high-level data analysis Python package.

It's at a very early stage of development and it's main purpose is to quickly prototype
the implementation of analysis methods for a proper implementation in `GammaLib`_ and `ctools`_.

* Code: https://github.com/gammapy/tevpy
* Docs: https://tevpy.readthedocs.org/

Show me some code!
------------------

`tevpy` gives you easy access to some frequently used methods in TeV gamma-ray astronomy:

What's the statistical significance when 10 events have been observed with a known background level of 4.2
according to [LiMa1983]_?

   >>> from tevpy.stats import significance
   >>> significance(n_observed=10, mu_background=4.2, method='lima')
   2.3979181291475453

What's the differential gamma-ray flux and spectral index of the Crab nebula at 3 TeV
according to [Meyer2010]_?

  >>> from tevpy.spec import crab
  >>> energy = 3
  >>> crab.diff_flux(energy, ref='meyer')
  1.8993523278650278e-12
  >>> crab.spectral_index(energy, ref='meyer')
  2.6763224503600429

Using `tevpy`
-------------

All functionality is in subpackages (e.g. `tevpy.stats` or `tevpy.spec`) ...
browse their docs to see if it contains the methods you want.

.. toctree::
  :maxdepth: 1

  introduction
  background/index
  obs/index
  spectrum/index
  stats/index
  utils/index
  references

Contact
-------

Found a bug or missing feature?

Make an issue or pull request on `GitHub <https://github.com/gammapy/tevpy>`_. 


.. _GammaLib: http://gammalib.sourceforge.net
.. _ctools: http://cta.irap.omp.eu/ctools/

