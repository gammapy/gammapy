# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.table import Table, Column
import astropy.units as u
from gammapy.utils.scripts import get_parser, set_up_logging_from_args
from gammapy.stats import significance_on_off
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum.utils import CountsPredictor
from gammapy.scripts import CTAPerf

log = logging.getLogger(__name__)

__all__ = ['SensitivityEstimator']


class SensitivityEstimator(object):
    """Estimate differential sensitivity for the ON/OFF method, ie 1D analysis

    Parameters
    ----------
    irf : `~gammapy.scripts.CTAPerf`
        IRF object
    livetime : `~astropy.units.Quantity`
        Livetime (object with the units of time), e.g. 5*u.h
    slope : `float`, optional
        Index of the spectral shape (Power-law), should be positive (>0)
    alpha : `float`, optional
        On/OFF normalisation
    sigma : `float`, optional
        Minimum significance
    gamma_min : `float`, optional
        Minimum number of gamma-rays
    bkg_sys : `float`, optional
        Fraction of Background systematics relative to the number of ON counts
    random: : `int`, optional
        Number of random trial to derive the number of gamma-rays

    Examples
    --------

    Compute and plot a sensitivity curve for CTA::

        from gammapy.scripts import CTAPerf, SensitivityEstimator
        filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'
        irf = CTAPerf.read(filename)
        sens = SensitivityEstimator(irf=irf, livetime='5h')
        sens.run()
        sens.print_results()
        sens.plot()

    Further examples in :gp-extra-notebook:`cta_sensitivity` .

    Notes
    -----
    For the moment, only the differential point-like sensitivity is computed at a fixed offset.
    This class allows to determine for each reconstructed energy bin the flux associated to the number of gamma-ray
    events for which the significance is 'sigma', and being larger than 'gamma_min' and 'bkg_sys'% larger than the
    number of background events in the ON region

    TODO:

    - make the computation for any source size
    - compute the integral sensitivity
    - Add options to use different spectral shape?
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

        self.energy = None
        self.diff_sens = None

    def get_bkg(self, bkg_rate):
        """Return the Background rate for each energy bin

        Parameters
        ----------
        bkg_rate : `~gammapy.scripts.BgRateTable`
            Table of background rate

        Returns
        -------
        rate : `~numpy.ndarray`
            Return an array of background counts (bins in reconstructed energy)
        """
        bkg = bkg_rate.data.data * self.livetime
        return bkg.value

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
        Defines a function `significance_on_off(x, off, alpha) - self.sigma`
        and uses scipy.optimize.newton to find the `x` for which this function
        is zero.
        """

        def target_function(on, off, alpha):
            return significance_on_off(on, off, alpha, method='lima') - self.sigma

        from scipy.optimize import newton
        excess = np.zeros_like(bkg_counts)
        for energy_bin, bkg_count in enumerate(bkg_counts):
            # if the number of bg events is to small just return the predefined minimum
            if bkg_count < 1:
                excess[energy_bin] = self.gamma_min
                continue

            off = bkg_count / self.alpha
            # provide a proper start guess for the minimizer
            on = bkg_count + self.gamma_min
            e = newton(target_function, x0=on, args=(off, self.alpha))

            # excess is defined as the number of on events minues the number of background events
            excess[energy_bin] = e - bkg_count

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

    def run(self):
        """Run the algorithm to compute the differential sensitivity as explained in the document of the class.
        """
        # Creation of the spectral shape
        norm = 1 * u.Unit('cm-2 s-1 TeV-1')
        index = self.slope
        ref = 1 * u.TeV
        model = PowerLaw(index=index, amplitude=norm, reference=ref)

        # Get the bins in reconstructed  energy
        reco_energy = self.irf.bkg.energy

        # Start the computation
        bkg_counts = self.get_bkg(self.irf.bkg)
        if self.random < 1:
            excess_counts = self.get_excess(bkg_counts)
        else:
            ex = self.get_excess(np.random.poisson(bkg_counts))
            for ii in range(self.random-1):
                ex += self.get_excess(np.random.poisson(bkg_counts))
            excess_counts = ex / float(self.random)

        phi_0 = self.get_1TeV_differential_flux(excess_counts, model, self.irf.aeff, self.irf.rmf)
        energy = reco_energy.log_center()
        dnde_model = model.evaluate(energy=energy, index=index, amplitude=1, reference=ref)
        diff_flux = (phi_0 * dnde_model * energy ** 2).to('erg / (cm2 s)')

        self.energy = reco_energy.log_center()
        self.diff_sens = diff_flux

    @property
    def diff_sensi_table(self):
        """Differential sensitivity table (`~astropy.table.Table`)."""
        table = Table()
        table['ENERGY'] = Column(self.energy, unit=self.energy.unit,
                                 description='Reconstructed Energy')
        table['FLUX'] = Column(self.diff_sens, unit=self.diff_sens.unit,
                               description='Differential flux')
        return table

    @property
    def _ref_diff_sensi(self):
        """Reference differential sensitivity table (`~astropy.table.Table`)"""
        table = Table()
        energy = self.irf.sens.energy.log_center()
        table['ENERGY'] = Column(energy, unit=energy.unit,
                                 description='Reconstructed Energy')
        flux = self.irf.sens.data.data
        table['FLUX'] = Column(flux, unit=flux.unit,
                               description='Differential flux')
        return table

    def print_results(self):
        """Print results to the console."""
        log.info("** Sensitivity **")
        self.diff_sensi_table.pprint()

        if log.getEffectiveLevel() == 10:
            log.debug("** ROOT Sensitivity **")
            self._ref_diff_sensi.pprint()
            rel_diff = (self.diff_sensi_table['FLUX']-self._ref_diff_sensi['FLUX'])/self._ref_diff_sensi['FLUX']
            log.debug("** Relative Difference (ref=ROOT)**")
            log.debug(rel_diff)

    def plot(self, ax=None):
        """Plot the sensitivity curve."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        fig.canvas.set_window_title("Sensitivity")
        ax = ax or plt.gca()

        ax.plot(self.energy.value, self.diff_sens.value, color='red', label=r"        $\sigma$="+str(self.sigma)+" T="+\
                                str(self.livetime.to('h').value)+"h \n"+r"$\alpha$="+str(self.alpha)+ \
                                r" Syst$_{BKG}$="+str(self.bkg_sys*100)+"%"+r" $\gamma_{min}$="+str(self.gamma_min))

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel('Reco Energy [{}]'.format(self.energy.unit))
        ax.set_ylabel('Sensitivity [{}]'.format(self.diff_sens.unit))
        if log.getEffectiveLevel() == 10:
            self.irf.sens.plot(color='black', label="ROOT")

        plt.legend()
        return ax


def sensitivity_estimate_cli(args=None):
    """Sensitivity estimator command line interface.

    Uses `~gammapy.scripts.SensitivityEstimator`.
    """
    parser = get_parser(description=SensitivityEstimator.__doc__, function=SensitivityEstimator)
    parser.add_argument('irffile', type=str,
                        help='IRF file (containing the path)')
    parser.add_argument('livetime', type=float,
                        help='Livetime in hours (units in u.h)')
    parser.add_argument('-slope', type=float, default=2.,
                        help='Slope of the power law (>0)')
    parser.add_argument('-alpha', type=float, default=0.2,
                        help='Optional: ON/OFF normalisation')
    parser.add_argument('-sigma', type=float, default=5.,
                        help='Optional: number of sigma for the sensitivity')
    parser.add_argument('-gamma_min', type=float, default=10.,
                        help='Optional: minimum number of gamma-rays')
    parser.add_argument('-bkg_sys', type=float, default=0.05,
                        help='Optional: Fraction of Background systematics relative to the number of ON counts')
    parser.add_argument('-nrand', type=int, default=0,
                        help='Optional: Number of random trial to derive the number of gamma-rays')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)
    set_up_logging_from_args(args)

    IRFs = CTAPerf.read(args.irffile)

    sensi = SensitivityEstimator(irf=IRFs,
                                 livetime=args.livetime * u.Unit('h'),
                                 slope=args.slope,
                                 alpha=args.alpha,
                                 sigma=args.sigma,
                                 gamma_min=args.gamma_min,
                                 bkg_sys=args.bkg_sys,
                                 random=args.nrand)
    sensi.run()
    sensi.print_results()
    import matplotlib.pyplot as plt
    sensi.plot()
    plt.show()


if __name__ == '__main__':
    sensitivity_estimate_cli()
