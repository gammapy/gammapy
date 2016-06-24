# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an in-development package for region handling based on Astropy.

The goal is to merge the functionality from pyregion and photutils apertures
and then after some time propose this package for inclusion in the Astropy core.

* Code : https://github.com/astropy/regions
* Docs : http://astropy-regions.readthedocs.io/en/latest/
"""

from .core import *
from .io import *
from .shapes import *
