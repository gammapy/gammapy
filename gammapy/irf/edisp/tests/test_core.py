# Licensed under a 3-clause BSD style license - see LICENSE.rst
from copy import deepcopy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.coordinates import Angle
from gammapy.irf import EnergyDispersion2D
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.testing import mpl_plot_check, requires_data


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

        migra_axis = MapAxis.from_bounds(
            0, 4, nbin=1000, node_type="edges", name="migra"
        )
        offset_axis = MapAxis.from_bounds(0, 2.5, nbin=5, unit="deg", name="offset")

        energy_true = energy_axis_true.edges[:-1]
        sigma = 0.15 / (energy_true / (1 * u.TeV)).value ** 0.3
        bias = 1e-3 * (energy_true - 1 * u.TeV).value

        cls.edisp2 = EnergyDispersion2D.from_gauss(
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            bias=bias,
            sigma=sigma,
            offset_axis=offset_axis,
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

        edisp_test = EnergyDispersion2D(axes=axes)
        with pytest.raises(ValueError) as error:
            wrong_unit = u.m**2
            EnergyDispersion2D(axes=axes, data=data * wrong_unit)
            assert error.match(
                f"Error: {wrong_unit} is not an allowed unit. {edisp_test.tag} "
                f"requires {edisp_test.default_unit} data quantities."
            )

        edisp = EnergyDispersion2D(axes=axes, data=data)

        hdu = edisp.to_table_hdu()
        energy = edisp.axes["energy_true"].edges
        assert_equal(hdu.data["ENERG_LO"][0], energy[:-1].value)
        assert hdu.header["TUNIT1"] == edisp.axes["energy_true"].unit

    def test_plot_migration(self):
        with mpl_plot_check():
            self.edisp.plot_migration()

    def test_plot_bias(self):
        with mpl_plot_check():
            self.edisp.plot_bias()

    def test_peek(self):
        with mpl_plot_check():
            self.edisp.peek()

    def test_eq(self):
        assert not self.edisp2 == self.edisp
        edisp1 = deepcopy(self.edisp)
        assert self.edisp == edisp1


@requires_data("gammapy-data")
def test_edisp2d_pointlike():
    filename = "$GAMMAPY_DATA/joint-crab/dl3/magic/run_05029748_DL3.fits"

    edisp = EnergyDispersion2D.read(filename)
    hdu = edisp.to_table_hdu()

    assert edisp.is_pointlike
    assert hdu.header["HDUCLAS3"] == "POINT-LIKE"
