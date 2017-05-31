#-*- coding: utf-8 -*-

"""
Compute the CTA sensitivity for a fixed observation config
Authors: J. Lefaucheur (LUTh), B. Khelifi (APC)
Date: 31/05/2017
Version: 0.1

Note: For the moment, only the differential point-like sensitivity is computed
ToDo: - make the computation for any source size
      - compute the integral sensitivity
      - Add options to use different spectral shape?

Inputs:
    - filename: str
        IRF filename (full path, in general there are here: ../datasets/cta/)
        e.g. ../datasets/cta/CTA-Performance-South-20170323/CTA-Performance-South-20deg-30m_20170323.fits
    - livetime: u.h
        Livetime (object with the units of time), e.g. 5*u.h
Optionnal inputs:
    - slope: float (>0)
        Index of the spectral shape (Power-law)
    - alpha: float
        On/OFF normalisation
    - sigma: float
        minimun significance
    - gamma_min: float
        Minimum number of gamma-rays
    - bkg_sys: float
        Fraction of Background systematics relative to the number of ON counts
    - verbosity: int
        level of print out [int: 0/1/2(debug)]. In the debug mode, stats for each bin are printed and the computed \
        sensitivity curve is superposed with the one stored in the IRF fits file

"""

# Inputs :

# IRFs :
#  - taux de fond en énergie reconstruite
#  - matrice de dispersion en énergie
#  - surface efficace en énergie vraie
# Paramètres :
#  - nsigma, alpha, temps, ngamma_min et systématiques fond

# Taux de fond, calcul de nombre de gamma

# calcul du flux avec la double intégrale et le nombre de gamma

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import sys

from gammapy.stats import significance_on_off
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum.utils import CountsPredictor
from gammapy.scripts import CTAPerf
from gammapy.utils.scripts import get_parser, set_up_logging_from_args


class cta_sensi_estim(object):
    """
    Parameters:
        - filename: str
            IRF filename (full path, in general there are here: ../datasets/cta/)
            e.g. ../datasets/cta/CTA-Performance-South-20170323/CTA-Performance-South-20deg-30m_20170323.fits
        - livetime: u.h
            Livetime (object with the units of time), e.g. 5*u.h
    Optional Parameters:
        - slope: float (>0)
            Index of the spectral shape (Power-law)
        - alpha: float
            On/OFF normalisation
        - sigma: float
            minimun significance
        - gamma_min: float
            Minimum number of gamma-rays
        - bkg_sys: float
            Fraction of Background systematics relative to the number of ON counts
        - verbosity: int
            level of print out [int: 0/1/2(debug)]. In the debug mode, stats for each bin are printed and the computed \
            sensitivity curve is superposed with the one stored in the IRF fits file
    """

    def __init__(self,irffile,livetime,slope=2.,alpha=0.2,sigma=5.,gamma_min=10.,bkg_sys=0.05, verbosity=0):
        self.filename = irffile
        self.livetime = (livetime*u.Unit('h')).to('s')
        self.slope = slope
        self.alpha = alpha
        self.sigma = sigma
        self.gamma_min = gamma_min
        self.bkg_sys = bkg_sys
        self.verbosity = verbosity
        self.energy = None
        self.diff_sens = None
        # Reading of the IRFs
        self.cta_perf = CTAPerf.read(self.filename)

    def get_bkg(self, bkg_rate):
        bkg = bkg_rate.data.data * self.livetime
        return bkg.value * u.Unit('')

    # Cf. np.vectorize for an time optimisation of this function
    # For the moment, search by dichotomy
    def get_excess(self, bkg_counts):

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
            #if self.verbosity == 2:
            #    print('# {}, {}'.format(start, stop))
            #    print('{}, {}'.format(idx, len(coarse_sigma)-1))
            if start == stop:
                print('LOGICAL ERROR> Impossible to find a number of gamma!')
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

            if self.verbosity > 0 :
                print('N_ex={}, N_fineEx={}, N_bkg={}, N_bkgsys={}, Sigma={}'.format(excess[icount],fine_excess[idx], \
                    bkg_counts[icount],self.bkg_sys * bkg_counts[icount],fine_sigma[idx]))

        return excess

    # Return differential fluxes
    def get_1TeV_differential_flux(self, excess_counts, sp_model, aeff, edisp):
        # Compute expected excess
        predictor = CountsPredictor(sp_model, aeff=aeff, edisp=edisp, livetime=self.livetime)
        predictor.run()
        counts = predictor.npred.data.data.value
        # Conversion in flux
        flux = excess_counts / counts * u.Unit('1 / (cm2 TeV s)')

        return flux

    def run(self):

        # Creation of the spectal shape
        norm = 1 * u.Unit('1 / (cm2 s TeV)')
        index = self.slope
        ref = 1 * u.TeV
        model = PowerLaw(index=index, amplitude=norm, reference=ref)

        # Get the bins in reconstructed  energy
        reco_energy = self.cta_perf.bkg.energy

        # Start the computation
        bkg_counts = self.get_bkg(self.cta_perf.bkg)
        excess_counts = self.get_excess(bkg_counts)

        Phi0 = self.get_1TeV_differential_flux(excess_counts, model, self.cta_perf.aeff, self.cta_perf.rmf)
        diff_flux = (Phi0 * model.evaluate(energy=reco_energy.log_center(), index=index, amplitude=1, reference=ref) * \
                     reco_energy.log_center() ** 2 * 1.60218).to('erg / (cm2 s)')

        self.energy = reco_energy.log_center()
        self.diff_sens = diff_flux

    # Plot the result
    def plot(self):

        ax = plt.gca()
        ax.plot(self.energy.value, self.diff_sens.value, color='red', label='GammaPy')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel('Reco Energy [{}]'.format(self.energy.unit))
        ax.set_ylabel('Sensitivity [{}]'.format(self.diff_sens.unit))
        if self.verbosity > 0:
            self.cta_perf.sens.plot(color='black', label="ROOT")
            plt.legend()
        # ax.errorbar(energy.value, values.value, xerr=xerr, fmt='o', **kwargs)

        print("\n #################################\n\033[34;1m   Emid        Sensi[{}]\033[0m".format(self.diff_sens.unit))
        for icount in range(len(self.energy)):
            print("{} {}".format(self.energy[icount].value, self.diff_sens[icount].value))
        print("\n")

        if self.verbosity > 0:
            print("\n #######################\n  REFERENCE \n   Emid         Sensi[erg/cm2/s]")
            for icount in range(len(self.cta_perf.sens.energy.log_center())):
                print("{} {}".format(self.cta_perf.sens.energy.log_center()[icount].value, \
                                     self.cta_perf.sens.data.data[icount].value))
            print("\n")

        plt.show()


def cta_sensitivity_main(args=None):

#    parser = get_parser(sensi_estim)
#    parser = argparse.ArgumentParser(description='Store the PSFs from Mc simulation in a 4D numpy table')
    parser = get_parser(description=cta_sensi_estim.__doc__, function=cta_sensi_estim)
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
    parser.add_argument("-verbosity", type=int, default=0,
                        help="Optional: verbose level [0/1/2(debug)]")
    args = parser.parse_args(args)
#    print(args)

    set_up_logging_from_args(args)


    sensi = cta_sensi_estim(irffile=args.irffile,
                            livetime=args.livetime,
                            slope=args.slope,
                            alpha=args.alpha,
                            sigma=args.sigma,
                            gamma_min=args.gamma_min,
                            bkg_sys=args.bkg_sys,
                            verbosity=args.verbosity)
    sensi.run()
    sensi.plot()


if __name__ == '__main__':
    cta_sensitivity_main()