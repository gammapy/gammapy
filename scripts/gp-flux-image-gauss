#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Simulate an image from a catalog and a reference image.
"""

from astropy.io import fits
import cmdl
from image.utils import empty_image
from image.simulate import to_image

# ------------------------------------------------------------
# Parse command line options
# ------------------------------------------------------------
option_list = cmdl.map_parameters
option_list += [cmdl.clobber, cmdl.verbose]
argument_list = ['catalog', 'image']
options = cmdl.parse(option_list, argument_list)

# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
image = empty_image(options.nxpix, options.nypix,
                    options.binsz, options.xref, options.yref,
                    options.proj, options.coordsys,
                    options.xrefpix, options.yrefpix,
                    options.dtype)
catalog = fits.open(options.catalog)[1].data
to_image(catalog, image)
image.writetofits(options.image, clobber=options.clobber)
