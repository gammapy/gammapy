# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import requires_data
from ...datasets import gammapy_extra
from ...image import SkyImage
from ..image_bin import image_bin_main


@requires_data('gammapy-extra')
def test_bin_image_main(tmpdir):
    """Run ``gammapy-bin-image`` and compare result to ``ctskymap``.
    """
    event_file = gammapy_extra.filename('test_datasets/irf/hess/pa/hess_events_023523.fits.gz')
    reference_file = gammapy_extra.filename('test_datasets/irf/hess/pa/ctskymap.fits.gz')
    out_file = str(tmpdir / 'gammapy_ctskymap.fits.gz')

    args = [event_file, reference_file, out_file]
    image_bin_main(args)

    actual = SkyImage.read(out_file)
    expected = SkyImage.read(reference_file)

    SkyImage.assert_allclose(actual, expected)
