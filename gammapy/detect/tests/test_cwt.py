# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from ...utils.testing import requires_dependency, requires_data
from ...detect import CWT, CWTKernels, CWTData
from ...maps import Map


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestCWT:
    """Test CWT algorithm."""

    def setup(self):
        filename = (
            "$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/counts.fits.gz"
        )
        image = Map.read(filename)
        background = image.copy(data=np.ones(image.data.shape, dtype=float))

        self.kernels = CWTKernels(n_scale=2, min_scale=3.0, step_scale=2.6, old=False)
        self.data = dict(image=image, background=background)
        self.cwt = CWT(
            kernels=self.kernels, significance_threshold=2., keep_history=True
        )

    def test_execute_iteration(self):
        cwt_data = CWTData(
            counts=self.data["image"],
            background=self.data["background"],
            n_scale=self.kernels.n_scale,
        )
        self.cwt._execute_iteration(data=cwt_data)
        residual = cwt_data.residual.data
        assert_allclose(residual.var(), 1.10209961137)
        assert_allclose(residual[100, 100], 4.60067024083)
        assert_allclose(residual[10, 10], 0.0210108511546)

    def test_transform(self):
        cwt_data = CWTData(
            counts=self.data["image"],
            background=self.data["background"],
            n_scale=self.kernels.n_scale,
        )
        self.cwt._transform(data=cwt_data)

        transform_3d = cwt_data.transform_3d.data
        assert_allclose(transform_3d[0, 100, 100], 0.0444647513236)
        assert_allclose(transform_3d[0, 10, 10], -0.00133091756454)
        assert_allclose(transform_3d[1, 100, 100], 0.00165322855919)
        assert_allclose(transform_3d[1, 10, 10], -9.2715980927e-05)

        error = cwt_data.error.data
        assert_allclose(error[0, 100, 100], 0.000840670230257)
        assert_allclose(error[0, 10, 10], 0.000810230288381)
        assert_allclose(error[1, 100, 100], 4.78305652411e-05)
        assert_allclose(error[1, 10, 10], 4.02498840476e-05)

        approx = cwt_data.approx.data
        assert_allclose(approx[100, 100], 0.353211779292)
        assert_allclose(approx[10, 10], -0.0210108511546)

        approx_bkg = cwt_data.approx_bkg.data
        assert_allclose(approx_bkg[100, 100], 0.99988318386)
        assert_allclose(approx_bkg[10, 10], 0.486747980289)

    def test_compute_support(self):
        cwt_data = CWTData(
            counts=self.data["image"],
            background=self.data["background"],
            n_scale=self.kernels.n_scale,
        )
        self.cwt._transform(data=cwt_data)
        self.cwt._compute_support(data=cwt_data)

        support_3d = cwt_data.support_3d.data
        assert_allclose(support_3d[0].sum(), 1095)
        assert_allclose(support_3d[1].sum(), 2368)

    def test_inverse_transform(self):
        cwt_data = CWTData(
            counts=self.data["image"],
            background=self.data["background"],
            n_scale=self.kernels.n_scale,
        )
        self.cwt._execute_iteration(data=cwt_data)

        model = cwt_data.model.data
        transform_2d = cwt_data.transform_2d.data
        assert_allclose(model.sum(), 11.7236771527)
        assert_allclose(transform_2d.sum(), 11.7236771527)

    def test_all_cwt_iterations(self):
        cwt_data = CWTData(
            counts=self.data["image"],
            background=self.data["background"],
            n_scale=self.kernels.n_scale,
        )
        self.cwt.analyze(data=cwt_data)

        transform_3d = cwt_data.transform_3d.data
        assert_allclose(transform_3d[0, 100, 100], 0.0401320295446)
        assert_allclose(transform_3d[0, 10, 10], -0.00117538066327)
        assert_allclose(transform_3d[1, 100, 100], 0.00112861578719)
        assert_allclose(transform_3d[1, 10, 10], -6.86491626269e-05)

        error = cwt_data.error.data
        assert_allclose(error[0, 100, 100], 0.00105275393856)
        assert_allclose(error[0, 10, 10], 0.000801986157367)
        assert_allclose(error[1, 100, 100], 5.54110995048e-05)
        assert_allclose(error[1, 10, 10], 3.9892257794e-05)

        approx = cwt_data.approx.data
        assert_allclose(approx[100, 100], 0.323369470219)
        assert_allclose(approx[10, 10], -0.0210240420041)

        approx_bkg = cwt_data.approx_bkg.data
        assert_allclose(approx_bkg[100, 100], 0.99988318386)
        assert_allclose(approx_bkg[10, 10], 0.486747980289)

        support_3d = cwt_data.support_3d.data
        assert_allclose(support_3d[0].sum(), 1151)
        assert_allclose(support_3d[1].sum(), 2368)

        model = cwt_data.model.data
        transform_2d = cwt_data.transform_2d.data
        assert_allclose(model.sum(), 103.96476418)
        assert_allclose(transform_2d.sum(), 9.91731463861)


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestCWTKernels:
    """Test CWTKernels"""

    def setup(self):
        self.kernels_new = CWTKernels(
            n_scale=2, min_scale=3.0, step_scale=2.6, old=False
        )
        self.kernels_old = CWTKernels(
            n_scale=2, min_scale=3.0, step_scale=2.6, old=True
        )

    def test_info(self):
        info_dict = self.kernels_new._info()
        assert_allclose(info_dict["Minimal scale"], self.kernels_new.min_scale)
        assert_allclose(info_dict["Kernels approx sum"], 0.99988318386)
        assert_allclose(info_dict["Kernels approx max"], 0.000386976177431)

    def test_cwt_kernels_new(self):
        assert_allclose(self.kernels_new.scales, [3., 7.8])
        assert_allclose(self.kernels_new.kern_approx.sum(), 0.99988318386)
        assert_allclose(self.kernels_new.kern_approx.max(), 0.000386976177431)
        assert_allclose(self.kernels_new.kern_base[0].sum(), 3.01663209714e-05)
        assert_allclose(self.kernels_new.kern_base[0].max(), 8.59947060958e-05)
        assert_allclose(self.kernels_new.kern_base[1].sum(), 4.84548284275e-06)
        assert_allclose(self.kernels_new.kern_base[1].max(), 1.88182106053e-06)

    def test_cwt_kernels_old(self):
        assert_allclose(self.kernels_old.scales, [3., 7.8])
        assert_allclose(self.kernels_old.kern_approx.sum(), 0.99988318386)
        assert_allclose(self.kernels_old.kern_approx.max(), 0.000386976177431)
        assert_allclose(self.kernels_old.kern_base[0].sum(), 0.000207093150419)
        assert_allclose(self.kernels_old.kern_base[0].max(), 0.0150679236063)
        assert_allclose(self.kernels_old.kern_base[1].sum(), 0.000146764160432)
        assert_allclose(self.kernels_old.kern_base[1].max(), 0.002228982782)

    def test_info_table(self):
        t = self.kernels_new.info_table
        assert_equal(t.colnames, ["Name", "Source"])
        assert_equal(len(t), 13)


@requires_dependency("scipy")
@requires_data("gammapy-extra")
class TestCWTData:
    """
    Test CWTData class.
    """

    def setup(self):
        filename = (
            "$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/counts.fits.gz"
        )
        image = Map.read(filename)
        background = image.copy(data=np.ones(image.data.shape, dtype=float))

        self.kernels = CWTKernels(n_scale=2, min_scale=3.0, step_scale=2.6, old=False)
        self.data = dict(image=image, background=background)
        self.cwt = CWT(
            kernels=self.kernels, significance_threshold=2., keep_history=True
        )
        self.cwt_data = CWTData(
            counts=image, background=background, n_scale=self.kernels.n_scale
        )
        self.cwt.analyze(data=self.cwt_data)

    def test_images(self):
        images = self.cwt_data.images()
        assert_allclose(images["counts"].data[25, 25], self.data["image"].data[25, 25])
        assert_allclose(
            images["background"].data[36, 63], self.data["background"].data[36, 63]
        )

        model_plus_approx = images["model_plus_approx"].data
        assert_allclose(model_plus_approx[100, 100], 0.753205544726)
        assert_allclose(model_plus_approx[10, 10], -0.0210240420041)

        maximal = images["maximal"].data
        assert_allclose(maximal[100, 100], 0.0401320295446)
        assert_allclose(maximal[10, 10], 0.)

        support_2d = images["support_2d"].data
        assert_allclose(support_2d.sum(), 2996)

    def test_cube_metrics_info(self):
        cubes = self.cwt_data.cubes()
        name = "transform_3d"
        cube = cubes[name].data
        info = self.cwt_data._metrics_info(data=cube, name=name)

        assert_equal(info["Shape"], "3D cube")
        assert_allclose(info["Variance"], 3.24405547338e-06)
        assert_allclose(info["Max value"], 0.041216412114)

    def test_image_info(self):
        t = self.cwt_data.image_info(name="residual")
        assert_equal(t.colnames, ["Metrics", "Source"])
        assert_equal(len(t), 7)

    def test_cube_info(self):
        t = self.cwt_data.cube_info(name="error")
        assert_equal(t.colnames, ["Metrics", "Source"])
        assert_equal(len(t), 7)

        t = self.cwt_data.cube_info(name="error", per_scale=True)
        assert_equal(t.colnames, ["Scale power", "Metrics", "Source"])
        assert_equal(len(t), 14)

    def test_info_table(self):
        t = self.cwt_data.info_table
        assert_equal(len(t.colnames), 7)
        assert_equal(len(t), 13)

    def test_sub(self):
        h = self.cwt.history
        diff = h[7] - h[5]
        assert_equal(diff.support_3d.data.sum(), 0)
        assert_allclose(diff.model.data.sum(), 20.4132906267)

    def test_io(self, tmpdir):
        filename = str(tmpdir / "test-cwt.fits")
        self.cwt_data.write(filename=filename, overwrite=True)
        approx = Map.read(filename, hdu="APPROX")
        assert_allclose(approx.data[100, 100], self.cwt_data._approx[100, 100])
        assert_allclose(approx.data[36, 63], self.cwt_data._approx[36, 63])

        transform_2d = Map.read(filename, hdu="TRANSFORM_2D")
        assert_allclose(
            transform_2d.data[100, 100], self.cwt_data.transform_2d.data[100, 100]
        )
        assert_allclose(
            transform_2d.data[36, 63], self.cwt_data.transform_2d.data[36, 63]
        )
