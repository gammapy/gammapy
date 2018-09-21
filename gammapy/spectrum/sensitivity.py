# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.table import Table, Column
import astropy.units as u
from gammapy.stats import excess_matching_significance_on_off
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum.utils import CountsPredictor

__all__ = ["SensitivityEstimator"]


class SensitivityEstimator(object):
    """Estimate differential sensitivity.

    Uses a 1D spectral analysis and on / off measurement.

    Parameters
    ----------
    irf : `~gammapy.scripts.CTAPerf`
        IRF object
    livetime : `~astropy.units.Quantity`
        Livetime (object with the units of time), e.g. 5*u.h
    slope : float, optional
        Index of the spectral shape (Power-law), should be positive (>0)
    alpha : float, optional
        On/OFF normalisation
    sigma : float, optional
        Minimum significance
    gamma_min : float, optional
        Minimum number of gamma-rays
    bkg_sys : float, optional
        Fraction of Background systematics relative to the number of ON counts

    Examples
    --------

    Compute and plot a sensitivity curve for CTA::

        from gammapy.irf import CTAPerf
        from gammapy.spectrum import SensitivityEstimator

        filename = '$GAMMAPY_DATA/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
        irf = CTAPerf.read(filename)
        sensitivity_estimator = SensitivityEstimator(irf=irf, livetime='5h')
        sensitivity_estimator.run()
        print(sensitivity_estimator.results_table)

    Further examples in :gp-extra-notebook:`cta_sensitivity` .

    Notes
    -----
    For the moment, only the differential point-like sensitivity is computed at a fixed offset.
    This class allows to determine for each reconstructed energy bin the flux associated to the number of gamma-ray
    events for which the significance is ``sigma``, and being larger than ``gamma_min`` and ``bkg_sys`` percent larger than the
    number of background events in the ON region.
    """

    def __init__(
        self, irf, livetime, slope=2., alpha=0.2, sigma=5., gamma_min=10., bkg_sys=0.05
    ):
        self.irf = irf
        self.livetime = u.Quantity(livetime).to("s")
        self.slope = slope
        self.alpha = alpha
        self.sigma = sigma
        self.gamma_min = gamma_min
        self.bkg_sys = bkg_sys

        self._results_table = None

    @property
    def results_table(self):
        """Results table (`~astropy.table.Table`)."""
        return self._results_table

    def run(self):
        """Run the computation."""

        # TODO: let the user decide on energy binning
        # then integrate bkg model and gamma over those energy bins.
        energy = self.irf.bkg.energy.log_center()

        bkg_counts = (self.irf.bkg.data.data * self.livetime).value

        excess_counts = excess_matching_significance_on_off(
            n_off=bkg_counts / self.alpha, alpha=self.alpha, significance=self.sigma
        )
        is_gamma_limited = excess_counts < self.gamma_min
        excess_counts[is_gamma_limited] = self.gamma_min

        model = PowerLaw(
            index=self.slope,
            amplitude=1 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1 * u.TeV,
        )

        # TODO: simplify the following computation
        predictor = CountsPredictor(
            model, aeff=self.irf.aeff, edisp=self.irf.rmf, livetime=self.livetime
        )
        predictor.run()
        counts = predictor.npred.data.data.value
        phi_0 = excess_counts / counts * u.Unit("cm-2 s-1 TeV-1")
        # TODO: should use model.__call__ here
        dnde_model = model.evaluate(
            energy=energy, index=self.slope, amplitude=1, reference=1 * u.TeV
        )
        diff_flux = (phi_0 * dnde_model * energy ** 2).to("erg / (cm2 s)")

        # TODO: take self.bkg_sys into account
        # and add a criterion 'bkg sys'
        criterion = []
        for idx in range(len(energy)):
            if is_gamma_limited[idx]:
                c = "gamma"
            else:
                c = "significance"
            criterion.append(c)

        table = Table(
            [
                Column(
                    data=energy,
                    name="energy",
                    format="5g",
                    description="Reconstructed Energy",
                ),
                Column(
                    data=diff_flux,
                    name="e2dnde",
                    format="5g",
                    description="Energy squared times differential flux",
                ),
                Column(
                    data=excess_counts,
                    name="excess",
                    format="5g",
                    description="Number of excess counts in the bin",
                ),
                Column(
                    data=bkg_counts,
                    name="background",
                    format="5g",
                    description="Number of background counts in the bin",
                ),
                Column(
                    data=criterion,
                    name="criterion",
                    description="Sensitivity-limiting criterion",
                ),
            ]
        )
        self._results_table = table
