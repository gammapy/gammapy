# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from ..utils.fits import table_from_row_data
from ..data import ObservationStats
from ..irf import EffectiveAreaTable, EnergyDispersion
from .core import CountsSpectrum, PHACountsSpectrum
from .utils import calculate_predicted_counts

__all__ = [
    'SpectrumStats',
    'SpectrumObservation',
    'SpectrumObservationList',
]


class SpectrumStats(ObservationStats):
    """Spectrum stats.

    Extends `~gammapy.data.ObservationStats` with spectrum
    specific information (energy bin info at the moment).
    """

    def __init__(self, **kwargs):
        self.energy_min = kwargs.pop('energy_min', None)
        self.energy_max = kwargs.pop('energy_max', None)
        super(SpectrumStats, self).__init__(**kwargs)

    def to_dict(self):
        """TODO: document"""
        data = super(SpectrumStats, self).to_dict()
        data['energy_min'] = self.energy_min
        data['energy_max'] = self.energy_max
        return data


class SpectrumObservation(object):
    """1D spectral analysis storage class

    This container holds the ingredients for 1D region based spectral analysis
    TODO: describe PHA, ARF, etc.

    Meta data is stored in the ``on_vector`` attribute. This reflects the OGIP
    convention.

    Parameters
    ----------
    on_vector : `~gammapy.spectrum.PHACountsSpectrum`
        On vector
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective Area
    off_vector : `~gammapy.spectrum.PHACountsSpectrum`, optional
        Off vector
    edisp : `~gammapy.irf.EnergyDispersion`, optional
        Energy dispersion matrix

    Examples
    --------
    .. plot::
        :include-source:

        from gammapy.spectrum import SpectrumObservation
        from gammapy.datasets import gammapy_extra
        import matplotlib.pyplot as plt

        phafile = gammapy_extra.filename('datasets/hess-crab4_pha/pha_obs23523.fits')
        obs = SpectrumObservation.read(phafile)
        obs.peek()
        plt.show()
    """

    def __init__(self, on_vector, aeff, off_vector=None, edisp=None):
        self.on_vector = on_vector
        self.aeff = aeff
        self.off_vector = off_vector
        self.edisp = edisp
        # TODO: Handle this in PHACountsSpectrum __init__ method
        if edisp is None:
            self.on_vector.rmffile = None
        if off_vector is None:
            self.on_vector.bkgfile = None

    @property
    def obs_id(self):
        """Unique identifier"""
        return self.on_vector.obs_id

    @property
    def livetime(self):
        """Dead-time corrected observation time"""
        return self.on_vector.livetime

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions"""
        return self.on_vector.backscal / self.off_vector.backscal

    @property
    def ebounds(self):
        """Energy bounds array."""
        return self.on_vector.energy.data.to('TeV')

    @property
    def lo_threshold(self):
        """Low energy threshold"""
        return self.on_vector.lo_threshold

    @property
    def hi_threshold(self):
        """High energy threshold"""
        return self.on_vector.hi_threshold

    @property
    def background_vector(self):
        """Background `~gammapy.spectrum.CountsSpectrum`

        bkg = alpha * n_off

        If alpha is a function of energy this will differ from
        self.on_vector * self.total_stats.alpha because the latter returns an
        average value for alpha.
        """
        energy = self.off_vector.energy
        data = self.off_vector.data * self.alpha
        return CountsSpectrum(data=data, energy=energy)

    @property
    def total_stats(self):
        """Return `~gammapy.spectrum.SpectrumStats`

        ``a_on`` and ``a_off`` are averaged over all energies.
        """
        return SpectrumStats(
            energy_min=self.ebounds[:-1],
            energy_max=self.ebounds[1:],
            n_on=int(self.on_vector.total_counts.value),
            n_off=int(self.off_vector.total_counts.value),
            a_on=np.mean(self.on_vector.backscal),
            a_off=np.mean(self.off_vector.backscal),
            obs_id=self.obs_id,
            livetime=self.livetime,
        )

    def stats(self, idx):
        """Compute stats for one energy bin.

        Parameters
        ----------
        idx : int
            Energy bin index

        Returns
        -------
        stats : `~gammapy.spectrum.SpectrumStats`
            Stats
        """
        return SpectrumStats(
            energy_min=self.ebounds[idx],
            energy_max=self.ebounds[idx + 1],
            n_on=int(self.on_vector.data.value[idx]),
            n_off=int(self.off_vector.data.value[idx]),
            a_on=self.on_vector.backscal[idx],
            a_off=self.off_vector.backscal[idx],
            obs_id=self.obs_id,
            livetime=self.livetime,
        )

    def stats_table(self):
        """Per-bin stats as a table.

        Returns
        -------
        table : `~astropy.table.Table`
            Table with stats for one energy bin in one row.
        """
        rows = [self.stats(idx).to_dict() for idx in range(len(self.ebounds) - 1)]
        return table_from_row_data(rows=rows)

    def predicted_counts(self, model):
        """Calculated npred given a model

        Parameters
        ----------
        model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model

        Returns
        -------
        npred : `~gammapy.spectrum.CountsSpectrum`
            Predicted counts
        """
        return calculate_predicted_counts(model=model,
                                          edisp=self.edisp,
                                          aeff=self.aeff,
                                          livetime=self.livetime)

    @classmethod
    def read(cls, filename):
        """Read `~gammapy.spectrum.SpectrumObservation` from OGIP files.

        BKG file, ARF, and RMF must be set in the PHA header and be present in
        the same folder.

        Parameters
        ----------
        filename : str
            OGIP PHA file to read
        """
        filename = make_path(filename)
        dirname = filename.parent
        on_vector = PHACountsSpectrum.read(filename)
        rmf, arf, bkg = on_vector.rmffile, on_vector.arffile, on_vector.bkgfile
        try:
            energy_dispersion = EnergyDispersion.read(str(dirname / rmf))
        except IOError:
            # TODO : Add logger and echo warning
            energy_dispersion = None
        try:
            off_vector = PHACountsSpectrum.read(str(dirname / bkg))
        except IOError:
            # TODO : Add logger and echo warning
            off_vector = None

        effective_area = EffectiveAreaTable.read(str(dirname / arf))
        return cls(on_vector=on_vector,
                   aeff=effective_area,
                   off_vector=off_vector,
                   edisp=energy_dispersion)

    def write(self, outdir=None, use_sherpa=False, overwrite=True):
        """Write OGIP files

        If you want to use the written files with Sherpa you have to set the
        ``use_sherpa`` flag. Then all files will be written in units 'keV' and
        'cm2'.

        Parameters
        ----------
        outdir : `~gammapy.extern.pathlib.Path`
            output directory, default: pwd
        use_sherpa : bool, optional
            Write Sherpa compliant files, default: False
        overwrite : bool, optional
            Overwrite, default: True
        """

        outdir = Path.cwd() if outdir is None else Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        phafile = self.on_vector.phafile
        bkgfile = self.on_vector.bkgfile
        arffile = self.on_vector.arffile
        rmffile = self.on_vector.rmffile

        # Write in keV and cm2 for sherpa
        if use_sherpa:
            self.on_vector.energy.data = self.on_vector.energy.data.to('keV')
            self.aeff.energy.data = self.aeff.energy.data.to('keV')
            self.aeff.data = self.aeff.data.to('cm2')
            if self.off_vector is not None:
                self.off_vector.energy.data = self.off_vector.energy.data.to('keV')
            if self.edisp is not None:
                self.edisp.e_reco.data = self.edisp.e_reco.data.to('keV')
                self.edisp.e_true.data = self.edisp.e_true.data.to('keV')
                # Set data to itself to trigger reset of the interpolator
                # TODO: Make NDData notice change of axis
                self.edisp.data = self.edisp.data

        self.on_vector.write(outdir / phafile, clobber=overwrite)
        self.aeff.write(outdir / arffile, clobber=overwrite)
        if self.off_vector is not None:
            self.off_vector.write(outdir / bkgfile, clobber=overwrite)
            if self.edisp is not None:
                self.edisp.write(str(outdir / rmffile), clobber=overwrite)

    def peek(self, figsize=(15, 15)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt
        plt.style.use('ggplot')

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        ax1.set_title('Counts')
        energy_unit = 'TeV'
        if self.off_vector is not None:
            self.background_vector.plot_hist(ax=ax1,
                                             label='alpha * n_off',
                                             color='darkblue',
                                             energy_unit=energy_unit)
        self.on_vector.plot_hist(ax=ax1,
                                 label='n_on',
                                 color='darkred',
                                 energy_unit=energy_unit,
                                 show_energy=(self.hi_threshold, self.lo_threshold))
        ax1.set_xlim(0.7 * self.lo_threshold.to(energy_unit).value,
                     1.3 * self.hi_threshold.to(energy_unit).value)
        ax1.legend(numpoints=1)

        ax2.set_title('Effective Area')
        e_unit = self.aeff.energy.unit
        self.aeff.plot(ax=ax2,
                       show_energy=(self.hi_threshold, self.lo_threshold))
        ax2.set_xlim(0.7 * self.lo_threshold.to(e_unit).value,
                     1.3 * self.hi_threshold.to(e_unit).value)

        ax3.axis('off')
        if self.off_vector is not None:
            ax3.text(0, 0.3, '{}'.format(self.total_stats), fontsize=18)

        ax4.set_title('Energy Dispersion')
        if self.edisp is not None:
            self.edisp.plot_matrix(ax=ax4)

        # TODO: optimize layout
        # plt.subplots_adjust(hspace = .2, left=.1)
        return fig

    def to_sherpa(self):
        """Create a `~sherpa.astro.data.DataPHA`

        associated background vectors and IRFs are also translated to sherpa
        objects and appended to the PHA instance
        """
        pha = self.on_vector.to_sherpa(name='pha_obs{}'.format(self.obs_id))
        arf = self.aeff.to_sherpa(name='arf_obs{}'.format(self.obs_id))
        if self.edisp is not None:
            rmf = self.edisp.to_sherpa(name='rmf_obs{}'.format(self.obs_id))
        else:
            rmf = None

        pha.set_response(arf, rmf)

        if self.off_vector is not None:
            bkg = self.off_vector.to_sherpa(name='bkg_obs{}'.format(self.obs_id))
            bkg.set_response(arf, rmf)
            pha.set_background(bkg, 1)

        # see https://github.com/sherpa/sherpa/blob/36c1f9dabb3350b64d6f54ab627f15c862ee4280/sherpa/astro/data.py#L1400
        pha._set_initial_quantity()
        return pha

    def __str__(self):
        """String representation"""
        ss = self.total_stats.__str__()
        return ss

    def _check_binning(self, **kwargs):
        """Check that ARF and RMF binnings are compatible
        """
        raise NotImplementedError

    @classmethod
    def stack(cls, obs_list, group_id=None):
        r"""Stack `~gammapy.spectrum.SpectrumObservationList`

        The stacking of :math:`j` observations is implemented as follows.
        :math:`k` and :math:`l` denote a bin in reconstructed and true energy,
        respectively. 

        .. math:: 

            \epsilon_{jk} =\left\{\begin{array}{cl} 1, & \mbox{if
                bin k is inside the energy thresholds}\\ 0, & \mbox{otherwise} \end{array}\right.

            \overline{\mathrm{n_{on}}}_k = \sum_{j} \mathrm{n_{on}}_{jk} \cdot
                \epsilon_{jk} 

            \overline{\mathrm{n_{off}}}_k = \sum_{j} \mathrm{n_{off}}_{jk} \cdot
                \epsilon_{jk} 

            \overline{\alpha}_k = \frac{\sum_{j}\alpha_{jk} \cdot
                \mathrm{n_{off}}_{jk} \cdot \epsilon_{jk}}{\overline{\mathrm {n_{off}}}}

            \overline{t} = \sum_{j} t_i

            \overline{\mathrm{aeff}}_l = \frac{\sum_{j}\mathrm{aeff}_{jl} 
                \cdot t_j}{\overline{t}}

            \overline{\mathrm{edisp}}_{kl} = \frac{\sum_{j} \mathrm{edisp}_{jkl} 
                \cdot \mathrm{aeff}_{jl} \cdot t_j \cdot \epsilon_{jk}}{\sum_{j} \mathrm{aeff}_{jl}
                \cdot t_j}


        Parameters
        ----------
        obs_list : `~gammapy.spectrum.SpectrumObservationList`
            Observations to stack
        group_id : int, optional
            ID for stacked observations
        """
        group_id = group_id or obs_list[0].obs_id

        # np.sum does not work with Quantities
        e_true = obs_list[0].aeff.energy
        e_reco = obs_list[0].on_vector.energy
        reco_bins = obs_list[0].on_vector.energy.nbins
        stacked_livetime = Quantity(0, 's')
        stacked_on_counts = np.zeros(e_reco.nbins)
        stacked_off_counts = np.zeros(e_reco.nbins)
        aefft = Quantity(np.zeros(e_true.nbins), 'cm2 s')
        aefftedisp = Quantity(np.zeros(shape=(e_reco.nbins, e_true.nbins)), 'cm2 s')
        backscal_on = np.zeros(e_reco.nbins)
        backscal_off = np.zeros(e_reco.nbins)
        lo_thresholds = list()
        hi_thresholds = list()

        for o in obs_list:
            stacked_livetime += o.livetime
            lo_thresholds.append(o.lo_threshold)
            hi_thresholds.append(o.hi_threshold)

            # Counts within safe range
            on_data = o.on_vector.data.copy()
            on_data[np.nonzero(o.on_vector.quality)] = 0
            stacked_on_counts += on_data
            off_data = o.off_vector.data.copy()
            off_data[np.nonzero(o.off_vector.quality)] = 0
            stacked_off_counts += off_data

            # Alpha
            backscal_on_data = o.on_vector.backscal.copy()
            backscal_on_data[np.nonzero(o.on_vector.quality)] = 0
            backscal_on += backscal_on_data * o.off_vector.data

            backscal_off_data = o.off_vector.backscal.copy()
            backscal_off_data[np.nonzero(o.off_vector.quality)] = 0
            backscal_off += backscal_off_data * o.off_vector.data

            # Exposure weighted IRFs
            aeff_data = o.aeff.evaluate(fill_nan=True)
            aefft_current = aeff_data * o.livetime
            aefft += aefft_current
            edisp_data = o.edisp.pdf_in_safe_range(o.lo_threshold, o.hi_threshold)
            aefftedisp += edisp_data.transpose() * aefft_current

        stacked_backscal_on = backscal_on / stacked_off_counts
        stacked_backscal_off = backscal_off / stacked_off_counts

        # there should be no nan values in backscal_on or backscal_off
        # this leads to problems when fitting the data
        alpha_correction = - 1
        idx = np.where(stacked_off_counts == 0)[0]
        stacked_backscal_on[idx] = alpha_correction
        stacked_backscal_off[idx] = alpha_correction

        stacked_aeff = aefft / stacked_livetime
        stacked_edisp = np.nan_to_num(aefftedisp / aefft)

        aeff = EffectiveAreaTable(energy=e_true,
                                  data=stacked_aeff.to('cm2'))
        edisp = EnergyDispersion(e_true=e_true,
                                 e_reco=e_reco,
                                 data=stacked_edisp.transpose())

        counts_kwargs = dict(
            lo_threshold=min(lo_thresholds),
            hi_threshold=max(hi_thresholds),
            livetime=stacked_livetime,
            obs_id=group_id,
            energy=e_reco,
        )

        on_vector = PHACountsSpectrum(backscal=stacked_backscal_on,
                                      data=stacked_on_counts,
                                      **counts_kwargs)
        off_vector = PHACountsSpectrum(backscal=stacked_backscal_off,
                                       data=stacked_off_counts,
                                       is_bkg=True,
                                       **counts_kwargs)

        return cls(on_vector=on_vector,
                   off_vector=off_vector,
                   edisp=edisp,
                   aeff=aeff)


class SpectrumObservationList(list):
    """
    List of `~gammapy.spectrum.SpectrumObservation`.
    """

    def obs(self, obs_id):
        """Return one observation

        Parameters
        ----------
        obs_id : int
            Identifier
        """
        obs_id_list = [o.obs_id for o in self]
        idx = obs_id_list.index(obs_id)
        return self[idx]

    @property
    def total_spectrum(self):
        """Stack all observations belongig to the list"""
        return SpectrumObservation.stack_observation_list(self)

    def info(self):
        """Info string"""
        ss = " *** SpectrumObservationList ***"
        ss += "\n\nNumber of observations: {}".format(len(self))
        ss += "\nObservation IDs: {}".format([o.obs_id for o in self])

        return ss

    def filter_by_reflected_regions(self, n_min):
        """Filter observation list according to number of reflected regions.

        Condition: number of reflected regions >= nmin.

        Parameters
        ----------
        n_min : int
            Minimum number of reflected regions

        Returns
        -------
        idx : `~np.array`
            Indices of element fulfilling the condition
        """
        val = [o.off_vector.meta.backscal for o in self]

        condition = np.array(val) >= n_min
        idx = np.nonzero(condition)
        return idx[0]

    def write(self, outdir=None, **kwargs):
        """Create OGIP files

        Parameters
        ----------
        outdir : str, `~gammapy.extern.pathlib.Path`, optional
            Output directory, default: pwd
        """
        for obs in self:
            obs.write(outdir=outdir, **kwargs)
