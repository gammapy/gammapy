# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ...utils.testing import requires_data, run_cli
from ...image import SkyImage
from ..main import cli


@requires_data('gammapy-extra')
def test_bin_image_main(tmpdir):
    """Run ``gammapy-bin-image`` and compare result to ``ctskymap``.
    """
    event_file = '$GAMMAPY_EXTRA/test_datasets/irf/hess/pa/hess_events_023523.fits.gz'
    reference_file = '$GAMMAPY_EXTRA/test_datasets/irf/hess/pa/ctskymap.fits.gz'
    out_file = str(tmpdir / 'gammapy_ctskymap.fits.gz')

    args = ['image', 'bin', event_file, reference_file, out_file]
    run_cli(cli, args)

    actual = SkyImage.read(out_file)
    expected = SkyImage.read(reference_file)
    SkyImage.assert_allclose(actual, expected)
