# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import absolute_import, division, print_function, unicode_literals
from ..stats import significance_on_off
from ..spectrum.models import PowerLaw
from ..spectrum.utils import CountsPredictor
from ..scripts import CTAPerf
from ..utils.scripts import get_parser, set_up_logging_from_args
from astropy.table import Table
import logging
import astropy.units as u
import numpy as np

log = logging.getLogger(__name__)

__all__ = ['SensiEstimator']

class SensiEstimator(object):
    """Compute the sensitivity at 0.5 degrees and XXX zenith angle

    note:: For the moment, only the differential point-like sensitivity is computed at a fixed offset
    todo:: - make the computation for any source size
          - compute the integral sensitivity
        - Add options to use different spectral shape?

    Parameters
    ----------
    irfobject: `~gammapy.scripts.CTAPerf`
        IRF object
    livetime: `~astropy.units.Quantity`
        Livetime (object with the units of time), e.g. 5*u.h
    slope: `float`, optional
        Index of the spectral shape (Power-law), should be positive (>0)
    alpha: `float`, optional
        On/OFF normalisation
    sigma: `float`, optional
        Minimum significance
    gamma_min: `float`, optional
        Minimum number of gamma-rays
    bkg_sys: `float`, optional
        Fraction of Background systematics relative to the number of ON counts

    Examples
    --------
        >>> import cta_sensitivity
        >>> from ..scripts import CTAPerf
        >>> irffile = '$GAMMAPY_EXTRA/datasets/cta/CTA-Performance-North-20170327/CTA-Performance-North-20deg-average-30m_20170327.fits'
        >>> ctairf = CTAPerf.read(irffile)
        >>> livetime = 0.5
        >>> sens = cta_sensitivity.SensiEstimator(irfobject=ctairf,livetime=livetime)
        >>> sens.run()
        >>> sens.plot()
        >>> sensi.print_results()

        Or

        :gp-extra-notebook:`cta_sensitivity`

    """

    def __init__(self,irfobject,livetime,slope=2.,alpha=0.2,sigma=5.,gamma_min=10.,bkg_sys=0.05):
        self.irf = irfobject
        self.livetime = (livetime*u.Unit('h')).to('s')
        self.slope = slope
        self.alpha = alpha
        self.sigma = sigma
        self.gamma_min = gamma_min
        self.bkg_sys = bkg_sys

        self.energy = None
        self.diff_sens = None

    def get_bkg(self, bkg_rate):
        """Return the Background rate for each energy bin

        Parameters
        ----------
        bkg_rate: `~gammapy.scripts.BgRateTable`
            Table of background rate

        Returns
        -------
        rate: `~numpy.array`
            Return an array of background rate (bins in reconstructed energy)
        """
        bkg = bkg_rate.data.data * self.livetime
        return bkg.value * u.Unit('')

    def get_excess(self, bkg_counts):
        """Compute the gamma-ray excess for each energy bin

        Parameters
        ----------
        bkg_counts: `~numpy.array`
            Array of background rate (bins in reconstructed energy)

        Returns
        -------
        count: `~numpy.array`
            Return an array of gamma-ray excess (bins in reconstructed energy)

        note:: For the moment, search by dichotomy
        todo:: Cf. np.vectorize for an time optimisation of this function
        """
        excess = np.zeros(len(bkg_counts))
        for icount in range(len(bkg_counts)):
    
            # Coarse search
            start, stop = -1., 6.
            coarse_excess = np.logspace(start=start, stop=stop, num=1000)
            coarse_on = coarse_excess + bkg_counts[icount]
            coarse_off = np.zeros(len(coarse_on)) + bkg_counts[icount] / self.alpha
            coarse_sigma = significance_on_off(n_on=coarse_on, n_off=coarse_off, alpha=self.alpha, method='lima')
            idx = np.abs(coarse_sigma - self.sigma).argmin()
        
            start = coarse_excess[max(idx-1,0)]
            stop = coarse_excess[min(idx+1, len(coarse_sigma)-1)]
            #log.debug('# {}, {}'.format(start, stop))
            #log.debug('{}, {}'.format(idx, len(coarse_sigma)-1))
            if start == stop:
                log.warning('LOGICAL ERROR> Impossible to find a number of gamma!')
                excess[icount] = -1
                continue
        
            # Finer search
            num = int((stop - start)/0.1)
            fine_excess = np.linspace(start=start, stop=stop, num=num)
            fine_on = fine_excess + bkg_counts[icount]
            fine_off = np.zeros(len(fine_on)) + bkg_counts[icount] / self.alpha
            fine_sigma = significance_on_off(n_on=fine_on, n_off=fine_off, alpha=self.alpha, method='lima')
            idx = np.abs(fine_sigma - self.sigma).argmin()
            if fine_excess[idx] >= self.gamma_min and fine_excess[idx] >= self.bkg_sys * bkg_counts[icount]:
                excess[icount] = fine_excess[idx]
            else:
                excess[icount] = max(self.gamma_min, self.bkg_sys * bkg_counts[icount])

            log.debug('N_ex={}, N_fineEx={}, N_bkg={}, N_bkgsys={}, Sigma={}'.format(excess[icount],fine_excess[idx], \
                bkg_counts[icount],self.bkg_sys * bkg_counts[icount],fine_sigma[idx]))

        return excess

    def get_1TeV_differential_flux(self, excess_counts, sp_model, aeff, edisp):
        """Compute the differential fluxes at 1 TeV

        Parameters
        ----------
        excess_counts: `~numpy.array`
            Array of gamma-ray excess (bins in reconstructed energy)
        sp_model: `spectrum.models.PowerLaw`
            Power Law object
        aeff: `~gammapy.irf.EffectiveAreaTable`
            Effective area
        edisp: `~gammapy.irf.EnergyDispersion2D`
            Energy dispersion

        Returns
        -------
        flux: `~astropy.units.Quantity`
            Return an array of 1TeV fluxes (bins in reconstructed energy)
        """

        # Compute expected excess
        predictor = CountsPredictor(sp_model, aeff=aeff, edisp=edisp, livetime=self.livetime)
        predictor.run()
        counts = predictor.npred.data.data.value

        # Conversion in flux
        flux = excess_counts / counts * u.Unit('1 / (cm2 TeV s)')

        return flux

    def run(self):
        """Do the computation
        """

        # Creation of the spectal shape
        norm = 1 * u.Unit('1 / (cm2 s TeV)')
        index = self.slope
        ref = 1 * u.TeV
        model = PowerLaw(index=index, amplitude=norm, reference=ref)

        # Get the bins in reconstructed  energy
        reco_energy = self.irf.bkg.energy

        # Start the computation
        bkg_counts = self.get_bkg(self.irf.bkg)
        excess_counts = self.get_excess(bkg_counts)

        Phi0 = self.get_1TeV_differential_flux(excess_counts, model, self.irf.aeff, self.irf.rmf)
        TeVtoErg = 1.60218
        diff_flux = (Phi0 * model.evaluate(energy=reco_energy.log_center(), index=index, amplitude=1, reference=ref) * \
                     reco_energy.log_center() ** 2 * TeVtoErg).to('erg / (cm2 s)')

        self.energy = reco_energy.log_center()
        self.diff_sens = diff_flux

    def plot(self):
        """Plot the result"""
        import matplotlib.pyplot as plt

        ax = plt.gca()
        ax.plot(self.energy.value, self.diff_sens.value, color='red', label='GammaPy')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel('Reco Energy [{}]'.format(self.energy.unit))
        ax.set_ylabel('Sensitivity [{}]'.format(self.diff_sens.unit))
        if log.getEffectiveLevel() == 10 :
            self.cta_perf.sens.plot(color='black', label="ROOT")
            plt.legend()
        plt.show()


    def print_results(self):
        """Print the result

        Returns
        =======
        array: `astropy.table.Table`
            Table containing [RecoEnergy, DiffFlux]
        """

        diff_sensi = Table([self.energy, self.diff_sens],\
                           names=('ENERGY', 'FLUX'), meta={'Energy': 'Differential flux'},\
                           dtype = ('float64', 'float64'))
        diff_sensi['ENERGY'].unit = self.energy.unit
        diff_sensi['FLUX'].unit = self.diff_sens.unit

        if log.getEffectiveLevel() == 10 :
            ref_diff_sensi = Table([self.irf.sens.energy.log_center(), self.irf.sens.data.data], \
                               names=('ENERGY', 'FLUX'), meta={'Energy': 'Differential flux'}, \
                               dtype=('float64', 'float64'))
            ref_diff_sensi['ENERGY'].unit = self.irf.sens.energy.log_center().unit
            ref_diff_sensi['FLUX'].unit = self.diff_sens.unit
            log.debug(ref_diff_sensi)

        return diff_sensi

def CTASensitivityMain(args=None):
    """Function to compute the CTA sensitivity at 0.5 degrees"""
    parser = get_parser(description=SensiEstimator.__doc__, function=SensiEstimator)
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
    parser.add_argument("-l", "--loglevel", default='info',
                    choices=['debug', 'info', 'warning', 'error', 'critical'],
                    help="Set the logging level")
    args = parser.parse_args(args)
    set_up_logging_from_args(args)


    IRFs = CTAPerf.read(args.filename)

    sensi = SensiEstimator( irfobject=IRFs,
                            livetime=args.livetime,
                            slope=args.slope,
                            alpha=args.alpha,
                            sigma=args.sigma,
                            gamma_min=args.gamma_min,
                            bkg_sys=args.bkg_sys)
    sensi.run()
    sensi.plot()
    sensi.print_results()


if __name__ == '__main__':
    CTASensitivityMain()
