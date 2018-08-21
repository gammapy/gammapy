# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.table import Table, Column
import astropy.units as u
from gammapy.stats import significance_on_off
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum.utils import CountsPredictor

log = logging.getLogger(__name__)

__all__ = ['SensitivityEstimator']


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
    random: : int, optional
        Number of random trial to derive the number of gamma-rays

    Examples
    --------

    Compute and plot a sensitivity curve for CTA::

        from gammapy.irf import CTAPerf
        from gammapy.spectrum import SensitivityEstimator

        filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
        irf = CTAPerf.read(filename)
        sens = SensitivityEstimator(irf=irf, livetime='5h')
        sens.run()
        sens.plot()

    Further examples in :gp-extra-notebook:`cta_sensitivity` .

    Notes
    -----
    For the moment, only the differential point-like sensitivity is computed at a fixed offset.
    This class allows to determine for each reconstructed energy bin the flux associated to the number of gamma-ray
    events for which the significance is ``sigma``, and being larger than ``gamma_min`` and ``bkg_sys`` percent larger than the
    number of background events in the ON region.
    """

    def __init__(self, irf, livetime, slope=2., alpha=0.2, sigma=5., gamma_min=10., bkg_sys=0.05, random=0):
        self.irf = irf

        self.livetime = u.Quantity(livetime).to('s')
        self.slope = slope
        self.alpha = alpha
        self.sigma = sigma
        self.gamma_min = gamma_min
        self.bkg_sys = bkg_sys
        self.random = random

        self._results_table = None

    @property
    def results_table(self):
        """Results table (`~astropy.table.Table`)."""
        return self._results_table

    def run(self):
        """Run the computation."""
        model = PowerLaw(
            index=self.slope,
            amplitude=1 * u.Unit('cm-2 s-1 TeV-1'),
            reference=1 * u.TeV,
        )

        reco_energy = self.irf.bkg.energy

        bkg_counts = (self.irf.bkg.data.data * self.livetime).value

        if self.random < 1:
            excess_counts = self.get_excess(bkg_counts)
        else:
            ex = self.get_excess(np.random.poisson(bkg_counts))
            for ii in range(self.random - 1):
                ex += self.get_excess(np.random.poisson(bkg_counts))
            excess_counts = ex / float(self.random)

        phi_0 = self.get_1TeV_differential_flux(excess_counts, model, self.irf.aeff, self.irf.rmf)
        energy = reco_energy.log_center()
        # TODO: should use model.__call__ here
        dnde_model = model.evaluate(energy=energy, index=self.slope, amplitude=1, reference=1 * u.TeV)
        diff_flux = (phi_0 * dnde_model * energy ** 2).to('erg / (cm2 s)')

        table = Table([
            Column(
                data=reco_energy.log_center(), name='ENERGY', format='5g',
                description='Reconstructed Energy',
            ),
            Column(
                data=diff_flux, name='FLUX', format='5g',
                description='Differential flux',
            ),
            Column(
                data=excess_counts, name='excess', format='5g',
                description='Number of excess counts in the bin',
            ),
            Column(
                data=bkg_counts, name='background', format='5g',
                description='Number of background counts in the bin',
            ),
        ])
        self._results_table = table

    def get_excess(self, bkg_counts):
        """Compute the gamma-ray excess for each energy bin.

        Parameters
        ----------
        bkg_counts : `~numpy.ndarray`
            Array of background counts (bins in reconstructed energy)

        Returns
        -------
        count : `~numpy.ndarray`
            Array of gamma-ray excess (bins in reconstructed energy)

        Notes
        -----
        Find the number of needed gamma excess events using newtons method.
        Defines a function ``significance_on_off(x, off, alpha) - self.sigma``
        and uses scipy.optimize.newton to find the `x` for which this function
        is zero.
        """
        from scipy.optimize import newton

        def target_function(on, off, alpha):
            return significance_on_off(on, off, alpha, method='lima') - self.sigma

        excess = np.zeros_like(bkg_counts)
        for energy_bin, bg_count in enumerate(bkg_counts):
            # if the number of bg events is to small just return the predefined minimum
            if bg_count < 1:
                excess[energy_bin] = self.gamma_min
                continue

            off = bg_count / self.alpha
            # provide a proper start guess for the minimizer
            on = bg_count + self.gamma_min
            e = newton(target_function, x0=on, args=(off, self.alpha))

            # excess is defined as the number of on events minues the number of background events
            excess[energy_bin] = e - bg_count

        return excess

    def get_1TeV_differential_flux(self, excess_counts, model, aeff, rmf):
        """Compute the differential fluxes at 1 TeV.

        Parameters
        ----------
        excess_counts : `~numpy.ndarray`
            Array of gamma-ray excess (bins in reconstructed energy)
        model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model
        aeff : `~gammapy.irf.EffectiveAreaTable`
            Effective area
        rmf : `~gammapy.irf.EnergyDispersion`
            RMF

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Array of 1TeV fluxes (bins in reconstructed energy)
        """
        # Compute expected excess
        predictor = CountsPredictor(model, aeff=aeff, edisp=rmf, livetime=self.livetime)
        predictor.run()
        counts = predictor.npred.data.data.value

        # Conversion in flux
        flux = excess_counts / counts * u.Unit('cm-2 s-1 TeV-1')

        return flux

    def plot(self, ax=None):
        """Plot the sensitivity curve."""
        import matplotlib.pyplot as plt

        energy = self.results_table['ENERGY'].quantity
        dnde = self.results_table['FLUX'].quantity

        fig = plt.figure()
        fig.canvas.set_window_title("Sensitivity")
        ax = ax or plt.gca()

        label = r"        $\sigma$=" + str(self.sigma) + " T=" + \
                str(self.livetime.to('h').value) + "h \n" + r"$\alpha$=" + str(self.alpha) + \
                r" Syst$_{BKG}$=" + str(self.bkg_sys * 100) + "%" + r" $\gamma_{min}$=" + str(self.gamma_min)
        ax.plot(energy.value, dnde.value, color='red', label=label)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel('Reco Energy [{}]'.format(energy.unit))
        ax.set_ylabel('Sensitivity [{}]'.format(dnde.unit))

        plt.legend()
        return ax
