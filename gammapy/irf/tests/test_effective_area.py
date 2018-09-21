# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
import astropy.units as u
from numpy.testing import assert_allclose, assert_equal
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency, requires_data, mpl_plot_check
from ...irf.effective_area import EffectiveAreaTable2D, EffectiveAreaTable


@pytest.fixture(scope="session")
def aeff():
    filename = (
        "$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )
    return EffectiveAreaTable2D.read(filename, hdu="AEFF")


class TestEffectiveAreaTable2D:

    # TODO: split this out into separate tests, especially the plotting
    # Add I/O test
    @staticmethod
    @requires_dependency("scipy")
    @requires_data("gammapy-extra")
    def test(aeff):
        assert aeff.data.axis("energy").nbins == 96
        assert aeff.data.axis("offset").nbins == 6
        assert aeff.data.data.shape == (96, 6)

        assert aeff.data.axis("energy").unit == "TeV"
        assert aeff.data.axis("offset").unit == "deg"
        assert aeff.data.data.unit == "m2"

        assert_quantity_allclose(aeff.high_threshold, 100 * u.TeV, rtol=1e-3)
        assert_quantity_allclose(aeff.low_threshold, 0.870964 * u.TeV, rtol=1e-3)

        test_val = aeff.data.evaluate(energy="14 TeV", offset="0.2 deg")
        assert_allclose(test_val.value, 683177.5, rtol=1e-3)

        # Test ARF export
        offset = 0.236 * u.deg
        e_axis = np.logspace(0, 1, 20) * u.TeV
        effareafrom2d = aeff.to_effective_area_table(offset, e_axis)

        energy = np.sqrt(e_axis[:-1] * e_axis[1:])
        area = aeff.data.evaluate(energy=energy, offset=offset)
        effarea1d = EffectiveAreaTable(
            energy_lo=e_axis[:-1], energy_hi=e_axis[1:], data=area
        )

        actual = effareafrom2d.data.evaluate(energy="2.34 TeV")
        desired = effarea1d.data.evaluate(energy="2.34 TeV")
        assert_equal(actual, desired)

        # Test ARF export #2
        offset = 1.2 * u.deg
        actual = aeff.to_effective_area_table(offset=offset).data.data
        desired = aeff.data.evaluate(offset=offset)
        assert_equal(actual.value, desired.value)

    @staticmethod
    @requires_dependency("matplotlib")
    @requires_data("gammapy-extra")
    def test_plot(aeff):
        with mpl_plot_check():
            aeff.plot()

        with mpl_plot_check():
            aeff.plot_energy_dependence()

        with mpl_plot_check():
            aeff.plot_offset_dependence()


class TestEffectiveAreaTable:
    @staticmethod
    @requires_dependency("scipy")
    @requires_dependency("matplotlib")
    @requires_data("gammapy-extra")
    def test_EffectiveAreaTable(tmpdir, aeff):
        arf = aeff.to_effective_area_table(offset=0.3 * u.deg)

        assert_quantity_allclose(arf.data.evaluate(), arf.data.data)

        with mpl_plot_check():
            arf.plot()

        filename = str(tmpdir / "effarea_test.fits")
        arf.write(filename)
        arf2 = EffectiveAreaTable.read(filename)

        assert_quantity_allclose(arf.data.evaluate(), arf2.data.evaluate())

        test_aeff = 0.6 * arf.max_area
        node_above = np.where(arf.data.data > test_aeff)[0][0]
        energy = arf.data.axis("energy")
        ener_above = energy.nodes[node_above]
        ener_below = energy.nodes[node_above - 1]
        test_ener = arf.find_energy(test_aeff)

        assert ener_below < test_ener and test_ener < ener_above

        elo_threshold = arf.find_energy(0.1 * arf.max_area)
        assert elo_threshold.unit == "TeV"
        assert_allclose(elo_threshold.value, 0.552741, rtol=1e-3)

        # Test evaluation outside safe range
        data = [np.nan, np.nan, 0, 0, 1, 2, 3, np.nan, np.nan]
        energy = np.logspace(0, 10, 10) * u.TeV
        aeff = EffectiveAreaTable(
            data=data, energy_lo=energy[:-1], energy_hi=energy[1:]
        )
        vals = aeff.evaluate_fill_nan()
        assert vals[1] == 0
        assert vals[-1] == 3

    @staticmethod
    def test_from_parametrization():
        # Log center of this is 100 GeV
        energy = [80, 125] * u.GeV
        area_ref = 1.65469579e+07 * u.cm ** 2

        area = EffectiveAreaTable.from_parametrization(energy, "HESS")

        assert_allclose(area.data.data, area_ref)
        assert area.data.data.unit == area_ref.unit

        # Log center of this is 0.1, 2 TeV
        energy = [0.08, 0.125, 32] * u.TeV
        area_ref = [1.65469579e+07, 1.46451957e+09] * u.cm * u.cm

        area = EffectiveAreaTable.from_parametrization(energy, "HESS")
        assert_allclose(area.data.data, area_ref)
        assert area.data.data.unit == area_ref.unit

        # TODO: Use this to test interpolation behaviour etc.

    @staticmethod
    def test_write():
        energy = np.logspace(0, 1, 11) * u.TeV
        energy_lo = energy[:-1]
        energy_hi = energy[1:]
        offset = np.linspace(0, 1, 4) * u.deg
        offset_lo = offset[:-1]
        offset_hi = offset[1:]
        data = np.ones(shape=(len(energy_lo), len(offset_lo))) * u.cm * u.cm

        aeff = EffectiveAreaTable2D(
            energy_lo=energy_lo,
            energy_hi=energy_hi,
            offset_lo=offset_lo,
            offset_hi=offset_hi,
            data=data,
        )
        hdu = aeff.to_fits()
        assert_equal(hdu.data["ENERG_LO"][0], aeff.data.axis("energy").lo.value)
        assert hdu.header["TUNIT1"] == aeff.data.axis("energy").lo.unit
