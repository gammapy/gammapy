# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.tests.helper import pytest, assert_quantity_allclose
from ...utils.testing import requires_data, requires_dependency
from ...utils.energy import EnergyBounds
from .. import CountsSpectrum


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestCountsSpectrum:

    def setup(self):
        self.counts = [0, 0, 2, 5, 17, 3] * u.ct
        self.bins = EnergyBounds.equal_log_spacing(1, 10, 6, 'TeV')
        self.spec = CountsSpectrum(data=self.counts, energy=self.bins)

    def test_wrong_init(self):
        bins = EnergyBounds.equal_log_spacing(1, 10, 7, 'TeV')
        with pytest.raises(ValueError):
            CountsSpectrum(data=self.counts, energy=bins)

    def test_evaluate(self):
        test_e = self.bins[2] + 0.1 * u.TeV
        test_eval = self.spec.evaluate(energy=test_e)
        assert_allclose(test_eval, self.counts[2])

    @requires_dependency('matplotlib')
    def test_plot(self):
        self.spec.plot()
        self.spec.plot_hist()

    def test_io(self, tmpdir):
        filename = tmpdir / 'test.fits'
        self.spec.write(filename)
        spec2 = CountsSpectrum.read(filename)
        assert_quantity_allclose(spec2.energy.data, self.bins)
