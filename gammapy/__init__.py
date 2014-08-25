# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Gammapy: A Python package for gamma-ray astronomy
=================================================

* Code: https://github.com/gammapy/gammapy
* Docs: https://gammapy.readthedocs.org/

The top-level `gammapy` namespace is almost empty,
it only contains this:

::

 test              --- Run Gammapy unit tests
 __version__       --- Gammapy version string


The Gammapy functionality is available for import from
the following sub-packages (e.g. `gammapy.spectrum`):

::

 `astro`          --- Astrophysical source and population models
 `background`     --- Background estimation and modeling
 `catalog`        --- Source catalog tools
 `data`           --- Data classes
 `datasets`       --- Access datasets
 `detect`         --- Source detection tools
 `image`          --- Image processing and analysis tools
 `irf`            --- Instrument response functions (IRFs)
 `morphology`     --- Morphology and PSF methods
 `obs`            --- Observation bookkeeping
 `shower`         --- Air shower image I/O and analysis
 `spectrum`       --- Spectrum estimation and modeling
 `stats`          --- Statistics tools
 `utils`          --- Utility functions and classes

"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *
# ----------------------------------------------------------------------------

# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    # put top-level package imports here
    pass
