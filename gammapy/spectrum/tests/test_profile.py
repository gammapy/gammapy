# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...utils.testing import requires_data, requires_dependency, mpl_plot_check
from ..profile import FluxPointProfiles


@requires_data("gammapy-extra")
class TestFluxPointProfiles:
    def setup_class(cls):
        path = "$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/binlike.fits"
        cls.profiles = FluxPointProfiles.read(path)

    def test_repr(self):
        s = repr(self.profiles)
        assert s == "FluxPointProfiles(n=24)"

    def test_get_profile(self):
        t = self.profiles.get_profile(0)
        assert len(t) == 21
        assert t.colnames == ['norm', 'dloglike']
        assert_allclose(t[0]['norm'], 2.2692, rtol=1e-3)
        assert_allclose(t[0]['dloglike'], 2124.5, rtol=1e-3)

    def test_interp_profile(self):
        norm = [2.2, 2.4, 2.7]
        t = self.profiles.interp_profile(0, norm)
        assert len(t) == 3
        assert t.colnames == ['norm', 'dloglike']
        assert_allclose(t['dloglike'], [2124.5451, 2128.8706, 2124.5451], rtol=1e-3)

    def test_get_reference_spectrum(self):
        s = self.profiles.get_reference_spectrum()
        assert len(s.energy) == 24
        assert s.energy.unit == 'MeV'
        assert_allclose(s.energy[-1].value, 86596.4, rtol=1e-3)
        assert s.values.unit == 'cm-2 s-1 MeV-1'
        assert_allclose(s.values[-1].value, 1.3335e-16, rtol=1e-3)

    @requires_dependency('matplotlib')
    def test_plot_sed(self):
        with mpl_plot_check():
            self.profiles.plot_sed()
