# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_equal, assert_almost_equal
from astropy.io import fits
from astropy.tests.helper import remote_data
from ...datasets import get_path
from ..bin_image import main as bin_image_main


@remote_data
def test_bin_image_main(tmpdir):
    """Run ``gammapy-bin-image`` and compare result to ``ctskymap``.
    """
    event_file = get_path('../test_datasets/irf/hess/pa/hess_events_023523.fits.gz', location='remote')
    reference_file = get_path('../test_datasets/irf/hess/pa/ctskymap.fits.gz', location='remote')
    out_file = str(tmpdir.join('gammapy_ctskymap.fits.gz'))
    bin_image_main([event_file, reference_file, out_file])

    gammapy_hdu = fits.open(out_file)[0]
    ctools_hdu = fits.open(reference_file)[0]
    assert_equal(gammapy_hdu.data, ctools_hdu.data)

    # I'm not sure if a header info comparison is useful here.
    # At the moment probably no ... the header is just copied
    # from the input to the output.
    # In the future we might implement similar sky and energy
    # selection options as `ctskymap` and then this could become
    # useful ...
    assert_almost_equal(gammapy_hdu.header['NAXIS1'], ctools_hdu.header['NAXIS1'])
