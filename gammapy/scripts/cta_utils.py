from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from ..spectrum import SpectrumObservation
from ..spectrum.utils import calculate_predicted_counts
from ..spectrum.core import PHACountsSpectrum
from ..utils.energy import EnergyBounds
from ..utils.random import get_random_state

from ..spectrum.models import AbsorbedSpectralModel, TableModel

__all__ = [
    'Target',
    'ObservationParameters',
    'CTASpectrumObservation',
]


class Target(object):
    """Class storing source information

    Parameters
    ----------
    name : `str`
        Name of the source
    model : `~gammapy.spectrum.models.SpectralModel`
        Model of the source
    redshift : `~astropy.units.Quantity`
        Redshift of the source
    ebl_model_name: `str`
        EBL model (franceschini, dominguez, finke or None)
    """

    def __init__(self, name=None,
                 model=None,
                 redshift=None,
                 ebl_model_name=None):
        self.name = name
        self.model = model
        self.redshift = redshift
        self.ebl_model_name = ebl_model_name
        self.abs_model = None
        filename = None
        if ebl_model_name in 'franceschini':
            filename = '$GAMMAPY_EXTRA/datasets/ebl/ebl_franceschini.fits.gz'
        elif ebl_model_name in 'dominguez':
            filename = '$GAMMAPY_EXTRA/datasets/ebl/ebl_dominguez11.fits.gz'
        elif ebl_model_name in 'finke':
            filename = '$GAMMAPY_EXTRA/datasets/ebl/ebl_dominguez11.fits.gz'
        else:
            print('No redshift?')

        if redshift is not None:
            absorption = TableModel.read_xspec_model(filename, redshift)
            self.abs_model = AbsorbedSpectralModel(spectral_model=model,
                                                   table_model=absorption)
        else:
            self.abs_model = model

    def __str__(self):
        """Target report (`str`)."""
        ss = '*** Target parameters ***\n'
        ss += 'Name={}\n'.format(self.name)
        for idx, param in enumerate(self.model.parameters):
            ss += '{}={}\n'.format(param, self.model.parameters[str(param)])
        ss += 'Redshift={}'.format(self.redshift)
        return ss

    def from_fermi_lat_catalogue(name):
        raise NotImplementedError


class ObservationParameters(object):
    """Class storing observation parameters

    Parameters
    ----------
    alpha : `~astropy.units.Quantity`
        Normalisation between ON and OFF regions
    livetime :  `~astropy.units.Quantity`
        Observation time
    offset :  `~astropy.units.Quantity`
        Offset of the source
    zenith :  `~astropy.units.Quantity`
        Zenith of the source
    emin :  `~astropy.units.Quantity`
        Minimal energy for simulation
    emax :  `~astropy.units.Quantity`
        Maximal energy for simulation
    """

    def __init__(self, alpha=None, livetime=None,
                 offset=None, zenith=None, emin=None,
                 emax=None):
        self.alpha = alpha
        self.livetime = livetime
        self.offset = offset
        self.zenith = zenith
        self.emin = emin
        self.emax = emax

    def __str__(self):
        """Observation summary report (`str`)."""
        ss = '*** Observation parameters summary ***\n'
        ss += 'alpha={} [{}]\n'.format(self.alpha.value, alpha.unit)
        ss += 'livetime={} [{}]\n'.format(self.livetime.value, livetime.unit)
        ss += 'offset={} [{}]\n'.format(self.offset.value, offset.unit)
        ss += 'zenith={} [{}]\n'.format(self.zenith.value, zenith.unit)
        ss += 'emin={} [{}]\n'.format(self.emin.value, emin.unit)
        ss += 'emax={} [{}]\n'.format(self.emax.value, emax.unit)
        return ss


class CTASpectrumObservation(object):
    """Class dedicated to simulate observation from one set
    of IRF and one target.

    TODO : Should be merge with `~gammapy.spectrum.SpectrumSimlation`

    Parameters
    ----------
    perf : `~gammapy.scripts.CTAPerf`
        CTA performance
    target : `~gammapy.scripts.Target`
        Source
    """

    def __init__(self, perf=None, target=None):
        self.perf = perf
        self.target = target
        self.simu = None
        self.on_vector = None
        self.off_vector = None

    def simulate_obs(self, obs_param):
        """
        Simulate observation with given parameters

        Parameters
        ----------
        obs_param : `~gammapy.scripts.ObservationParameters`
        """
        livetime = obs_param.livetime
        alpha = obs_param.alpha.value
        offset = obs_param.offset
        emin = obs_param.emin
        emax = obs_param.emax

        model = self.target.abs_model

        # Create energy dispersion
        e_reco_min = self.perf.bkg.energy.data[0]
        e_reco_max = self.perf.bkg.energy.data[-1]
        e_reco_bin = self.perf.bkg.energy.nbins
        e_reco_axis = EnergyBounds.equal_log_spacing(e_reco_min,
                                                     e_reco_max,
                                                     e_reco_bin,
                                                     'TeV')

        e_true_min = self.perf.aeff.energy.data[0]
        e_true_max = self.perf.aeff.energy.data[-1]
        e_true_bin = self.perf.aeff.energy.nbins
        e_true_axis = EnergyBounds.equal_log_spacing(e_true_min,
                                                     e_true_max,
                                                     e_true_bin,
                                                     'TeV')

        rmf = self.perf.edisp.to_energy_dispersion(offset,
                                                   e_reco=e_reco_axis,
                                                   e_true=e_true_axis)

        # Compute expected counts
        reco_energy = self.perf.bkg.energy
        bkg_rate_values = self.perf.bkg.data.data * livetime.to('s')
        predicted_counts = calculate_predicted_counts(model=model,
                                                      aeff=self.perf.aeff,
                                                      livetime=livetime,
                                                      edisp=rmf,
                                                      e_reco=e_reco_axis)

        # Randomise counts
        rand = get_random_state('random-seed')
        on_counts = rand.poisson(predicted_counts.data.data.value)  # excess
        bkg_counts = rand.poisson(bkg_rate_values.value)  # bkg in ON region
        off_counts = rand.poisson(
            bkg_rate_values.value / alpha)  # bkg in OFF region

        on_counts += bkg_counts  # evts in ON region

        # Create SpectrumObservation
        counts_kwargs = dict(energy=reco_energy,
                             livetime=livetime,
                             creator='gammapy')

        self.on_vector = PHACountsSpectrum(data=on_counts,
                                           backscal=1,
                                           **counts_kwargs)

        self.off_vector = PHACountsSpectrum(energy=reco_energy,
                                            data=off_counts,
                                            livetime=livetime,
                                            backscal=1. / alpha,
                                            is_bkg=True,
                                            creator='gammapy')

        obs = SpectrumObservation(on_vector=self.on_vector,
                                  off_vector=self.off_vector,
                                  aeff=self.perf.aeff,
                                  edisp=rmf)

        # Set threshold according to the closest energy reco from bkg bins
        idx_min = np.abs(reco_energy.data - emin).argmin()
        idx_max = np.abs(reco_energy.data - emax).argmin()
        obs.lo_threshold = reco_energy.data[idx_min]
        obs.hi_threshold = reco_energy.data[idx_max]

        # print('{},{}'.format())
        self.simu = obs

    def peek(self):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                       figsize=(10, 5))

        # Spectrum plot
        #energy_range = [self.simu.lo_threshold, self.simu.hi_threshold]
        energy_range = [0.01 * u.TeV, 100 * u.TeV]
        self.target.abs_model.plot(ax=ax1, energy_range=energy_range,
                                   label='Absorbed model')
        plt.text(0.55, 0.65, self.target.__str__(),
                 style='italic', transform=ax1.transAxes, fontsize=7,
                 bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
        ax1.set_xlim([energy_range[0].value, energy_range[1].value])
        ax1.set_ylim([1.e-15, 1.e-8])
        ax1.grid(which='both')
        ax1.legend(loc=0)

        # Counts plot
        on_off = self.on_vector.data.data.value
        off = 1. / self.off_vector.backscal * self.off_vector.data.data.value
        excess = on_off - off
        bins = self.on_vector.energy.data.value[:-1]
        x = self.on_vector.energy.nodes.value
        ax2.hist(x, bins=bins, weights=on_off,
                 facecolor='blue', alpha=1, label='ON')
        ax2.hist(x, bins=bins, weights=off,
                 facecolor='green', alpha=1, label='OFF')
        ax2.hist(x, bins=bins, weights=excess,
                 facecolor='red', alpha=1, label='EXCESS')
        ax2.legend(loc='best')
        ax2.set_xscale('log')
        ax2.set_xlabel('Energy [TeV]')
        ax2.set_ylabel('Expected counts')
        ax2.set_xlim([energy_range[0].value, energy_range[1].value])
        ax2.set_ylim([0.0001, on_off.max()*(1+0.05)])
        ax2.vlines(self.simu.lo_threshold.value, 0, 1.1 * on_off.max(),
                  linestyles='dashed')
        ax2.grid(which='both')
        plt.text(0.55, 0.05, self.simu.__str__(),
                 style='italic', transform=ax2.transAxes, fontsize=7,
                 bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
        plt.tight_layout()
        plt.show()
        return fig
