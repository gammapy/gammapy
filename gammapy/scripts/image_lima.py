# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import click
click.disable_unicode_literals_warning = True

import numpy as np
from astropy.io import fits
from astropy.convolution import Tophat2DKernel

from ..detect import compute_lima_map, compute_lima_on_off_map
from ..data import FitsMapBunch

__all__ = ['image_lima']

log = logging.getLogger(__name__)

@click.command()
@click.argument('infile')
@click.argument('outfile')
@click.option('--theta', default=0.1, help='On-region correlation radius (deg)')
@click.option('--onoff', is_flag=True, default=False, help='Compute Li&Ma maps for'
              'on/off observation.')
@click.option('--overwrite', is_flag=True, default=False, help='Overwrite existing output file?')
def image_lima(infile, outfile, theta, overwrite, onoff):
    """
    Compute Li&Ma significance maps for a given set of input maps.

    """
    log.info('Reading {0}'.format(infile))
    data = FitsMapBunch.read(infile)
    
    # Convert theta to pix
    theta_pix = theta / data._ref_header['CDELT2']
    kernel = Tophat2DKernel(theta_pix)
    with np.errstate(invalid='ignore', divide='ignore'):
        if not onoff:
            result = compute_lima_map(data.counts, data.background,
                                      data.exposure, kernel)
        else:
            result = compute_lima_on_off_map(data.On, data.Off, data.OnExposure,
                                             data.OffExposure, kernel)
    log.info('Computing derived maps')

    log.info('Writing {0}'.format(outfile))
    result.write(outfile, header=data._ref_header, clobber=overwrite)


