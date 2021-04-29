# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.irf import EDispKernel, EnergyDispersion2D
from gammapy.maps import MapAxis, MapAxes
from gammapy.utils.testing import mpl_plot_check, requires_data, requires_dependency


class TestEDispKernel:
    def setup(self):
        energy_axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=100)
        energy_axis_true = energy_axis.copy(name="energy_true")

        self.resolution = 0.1
        self.bias = 0
        self.edisp = EDispKernel.from_gauss(
            energy_axis_true=energy_axis_true,
            energy_axis=energy_axis,
            pdf_threshold=1e-7,
            sigma=self.resolution,
            bias=self.bias,
        )

    def test_from_diagonal_response(self):
        energy_axis_true = MapAxis.from_energy_edges([0.5, 1, 2, 4, 6] * u.TeV, name="energy_true")
        energy_axis = MapAxis.from_energy_edges([2, 4, 6] * u.TeV)

        edisp = EDispKernel.from_diagonal_response(energy_axis_true, energy_axis)

        assert edisp.pdf_matrix.shape == (4, 2)
        expected = [[0, 0], [0, 0], [1, 0], [0, 1]]

        assert_equal(edisp.pdf_matrix, expected)

        # Test square matrix
        edisp = EDispKernel.from_diagonal_response(energy_axis_true)
        assert_allclose(edisp.axes["energy"].edges, edisp.axes["energy_true"].edges)
        assert edisp.axes["energy"].unit == "TeV"
        assert_equal(edisp.pdf_matrix[0][0], 1)
        assert_equal(edisp.pdf_matrix[2][0], 0)
        assert edisp.pdf_matrix.sum() == 4

    def test_to_image(self):
        energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=3)
        energy_axis_true = MapAxis.from_energy_bounds(
            "0.08 TeV", "20 TeV", nbin=5, name="energy_true"
        )
        edisp = EDispKernel.from_gauss(
            energy_axis=energy_axis,
            energy_axis_true=energy_axis_true,
            sigma=0.2,
            bias=0.1
        )
        im = edisp.to_image()

        assert im.pdf_matrix.shape == (5, 1)
        assert_allclose(
            im.pdf_matrix, [[0.97142], [1.0], [1.0], [1.0], [0.12349]], rtol=1e-3
        )
        assert_allclose(im.axes["energy"].edges, [0.1, 10] * u.TeV)

    def test_str(self):
        assert "EDispKernel" in str(self.edisp)

    def test_evaluate(self):
        # Check for correct normalization
        pdf = self.edisp.evaluate(energy_true=3.34 * u.TeV)
        assert_allclose(np.sum(pdf), 1, atol=1e-2)

    def test_get_bias(self):
        bias = self.edisp.get_bias(3.34 * u.TeV)
        assert_allclose(bias, self.bias, atol=1e-2)

    def test_get_resolution(self):
        resolution = self.edisp.get_resolution(3.34 * u.TeV)
        assert_allclose(resolution, self.resolution, atol=1e-2)

    def test_io(self, tmp_path):
        indices = np.array([[1, 3, 6], [3, 3, 2]])
        desired = self.edisp.pdf_matrix[indices]
        self.edisp.write(tmp_path / "tmp.fits")
        edisp2 = EDispKernel.read(tmp_path / "tmp.fits")
        actual = edisp2.pdf_matrix[indices]
        assert_allclose(actual, desired)

    @requires_dependency("matplotlib")
    def test_plot_matrix(self):
        with mpl_plot_check():
            self.edisp.plot_matrix()

    @requires_dependency("matplotlib")
    def test_plot_bias(self):
        with mpl_plot_check():
            self.edisp.plot_bias()

    @requires_dependency("matplotlib")
    def test_peek(self):
        with mpl_plot_check():
            self.edisp.peek()


@requires_data()
class TestEnergyDispersion2D:
    @classmethod
    def setup_class(cls):
        filename = "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz"
        cls.edisp = EnergyDispersion2D.read(filename, hdu="EDISP")

        # Make a test case
        energy_axis_true = MapAxis.from_energy_bounds(
            "0.1 TeV", "100 TeV", nbin=50, name="energy_true"
        )

        migra_axis = MapAxis.from_bounds(0, 4, nbin=1000, node_type="edges", name="migra")
        offset_axis = MapAxis.from_bounds(0, 2.5, nbin=5, unit="deg", name="offset")

        energy_true = energy_axis_true.edges[:-1]
        sigma = 0.15 / (energy_true / (1 * u.TeV)).value ** 0.3
        bias = 1e-3 * (energy_true - 1 * u.TeV).value

        cls.edisp2 = EnergyDispersion2D.from_gauss(
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            bias=bias,
            sigma=sigma,
            offset_axis=offset_axis
        )

    def test_str(self):
        assert "EnergyDispersion2D" in str(self.edisp)

    def test_evaluation(self):
        # Check output shape
        energy = [1, 2] * u.TeV
        migra = np.array([0.98, 0.97, 0.7])
        offset = [0.1, 0.2, 0.3, 0.4] * u.deg
        actual = self.edisp.evaluate(
            energy_true=energy.reshape(-1, 1, 1),
            migra=migra.reshape(1, -1, 1),
            offset=offset.reshape(1, 1, -1),
        )
        assert_allclose(actual.shape, (2, 3, 4))

        # Check evaluation at all nodes
        actual = self.edisp.evaluate().shape
        desired = (
            self.edisp.axes["energy_true"].nbin,
            self.edisp.axes["migra"].nbin,
            self.edisp.axes["offset"].nbin,
        )
        assert_equal(actual, desired)

    def test_exporter(self):
        # Check RMF exporter
        offset = Angle(0.612, "deg")
        e_reco = MapAxis.from_energy_bounds(1, 10, 7, "TeV").edges
        e_true = MapAxis.from_energy_bounds(0.8, 5, 5, "TeV").edges
        rmf = self.edisp.to_edisp_kernel(offset, energy_true=e_true, energy=e_reco)
        assert_allclose(rmf.data.data[2, 3], 0.08, atol=5e-2)  # same tolerance as above

    def test_write(self):
        energy_axis_true = MapAxis.from_energy_bounds(
            "1 TeV", "10 TeV", nbin=10, name="energy_true"
        )

        offset_axis = MapAxis.from_bounds(
            0, 1, nbin=3, unit="deg", name="offset", node_type="edges"
        )

        migra_axis = MapAxis.from_bounds(0, 3, nbin=3, name="migra", node_type="edges")

        axes = MapAxes([energy_axis_true, migra_axis, offset_axis])

        data = np.ones(shape=axes.shape)
        edisp = EnergyDispersion2D(axes=axes, data=data)

        hdu = edisp.to_table_hdu()
        energy = edisp.axes["energy_true"].edges
        assert_equal(hdu.data["ENERG_LO"][0], energy[:-1].value)
        assert hdu.header["TUNIT1"] == edisp.axes["energy_true"].unit

    @requires_dependency("matplotlib")
    def test_plot_migration(self):
        with mpl_plot_check():
            self.edisp.plot_migration()

    @requires_dependency("matplotlib")
    def test_plot_bias(self):
        with mpl_plot_check():
            self.edisp.plot_bias()

    @requires_dependency("matplotlib")
    def test_peek(self):
        with mpl_plot_check():
            self.edisp.peek()


@requires_data("gammapy-data")
def test_get_bias_energy():
    """Obs read from file"""
    rmffile = "$GAMMAPY_DATA/joint-crab/spectra/hess/rmf_obs23523.fits"
    edisp = EDispKernel.read(rmffile)
    thresh_lo = edisp.get_bias_energy(0.1)
    assert_allclose(thresh_lo.to("TeV").value, 0.9174, rtol=1e-4)
