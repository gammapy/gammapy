# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import numpy as np
from ..hpx_utils import HPX
from ..fits_utils import write_fits_image
from ..skymap import HpxMap


def test_hpxmap(tmpdir):
    n = np.ones((10, 192), 'd')
    hpx = HPX(4, False, 'GAL')

    filename = str(tmpdir / 'test_hpx.fits')
    hpx.write_fits(n, filename, clobber=True)

    ebins = np.logspace(2, 5, 8)

    hpx_2 = HPX(1024, False, 'GAL', region='DISK(110.,75.,2.)', ebins=ebins)
    npixels = hpx_2.npix

    n2 = np.ndarray((8, npixels), 'd')
    for i in range(8):
        n2[i].flat = np.arange(npixels)

    hpx_map = HpxMap(n2, hpx_2)
    wcs, wcs_data = hpx_map.make_wcs_from_hpx(normalize=True)

    wcs_out = hpx_2.make_wcs(3)

    filename = str(tmpdir / 'test_hpx_2_wcs.fits')
    write_fits_image(wcs_data, wcs_out, filename)

    # TODO: add assert statements
