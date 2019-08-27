# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.maps import Map
from gammapy.scripts.main import cli
from gammapy.utils.testing import assert_wcs_allclose, requires_data, run_cli


@requires_data()
def test_bin_image_main(tmpdir):
    """Run ``gammapy-bin-image`` and compare result to ``ctskymap``.
    """
    event_file = "$GAMMAPY_DATA/tests/irf/hess/pa/hess_events_023523.fits.gz"
    reference_file = "$GAMMAPY_DATA/tests/irf/hess/pa/ctskymap.fits.gz"
    out_file = str(tmpdir / "gammapy_ctskymap.fits.gz")

    args = ["image", "bin", event_file, reference_file, out_file]
    run_cli(cli, args)

    actual = Map.read(out_file)
    expected = Map.read(reference_file)
    assert_allclose(actual.data, expected.data)
    assert_wcs_allclose(actual.geom.wcs, expected.geom.wcs)
