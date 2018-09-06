# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...utils.testing import requires_dependency, requires_data
from ..psf_3d import PSF3D
from ..psf_check import PSF3DChecker


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestPSF3DChecker:
    def setup(self):
        filename = "$GAMMAPY_EXTRA/datasets/hess-hap-hd-prod3/psf_table.fits.gz"
        self.psf = PSF3D.read(filename)

    def test_all(self):
        checker = PSF3DChecker(psf=self.psf)
        checker.check_all()
        res = checker.results

        # Make sure defaults don't change unnoticed
        conf = checker.config
        assert_allclose(conf["d_norm"], 0.01)
        assert_allclose(conf["containment_fraction"], 0.68)
        assert_allclose(conf["d_rel_containment"], 0.7)

        # Check results
        assert res["nan"]["status"] == "failed"
        assert res["nan"]["n_failed_bins"] == 4
        assert res["normalise"]["status"] == "ok"
        assert res["containment"]["status"] == "ok"
        assert res["status"] == "failed"
