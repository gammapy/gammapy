# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.utils import minversion

__all__ = [
    "COPY_IF_NEEDED",
]

# This is copied from astropy.utils.compat
NUMPY_LT_2_0 = not minversion(np, "2.0.0.dev")

COPY_IF_NEEDED = False if NUMPY_LT_2_0 else None
