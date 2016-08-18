# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from ..extern.pathlib import Path
from ..extern.bunch import Bunch
from ..utils.energy import EnergyBounds
from ..utils.scripts import make_path
from ..data import ObservationStats
from ..irf import EffectiveAreaTable, EnergyDispersion
from .core import CountsSpectrum, PHACountsSpectrum
from .utils import calculate_predicted_counts

__all__ = [
    'SpectrumObservation',
    'SpectrumObservationList',
]


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
        """
        energy = self.off_vector.energy
        data = self.off_vector.data * self.alpha
        return CountsSpectrum(data=data, energy=energy)

    @property
    def total_stats(self):
        """Return `~gammapy.data.ObservationStats`"""
        # TODO: Introduce SpectrumStats class inheriting from ObservationStats
        # in order to add spectrum specific information
        kwargs = dict(
            n_on=int(self.on_vector.total_counts.value),
            n_off=int(self.off_vector.total_counts.value),
            a_on=self.on_vector.backscal,
            a_off=self.off_vector.backscal,
            obs_id=self.obs_id,
            livetime=self.livetime,
        )
        return ObservationStats(**kwargs)

    def stats(self, nbin):
        """Return `~gammapy.data.ObservationStats` for one bin"""
        # TODO: Introduce SpectrumStats class inheriting from ObservationStats
        # in order to add spectrum specific information
        kwargs = dict(
            n_on=int(self.on_vector.data.value[nbin]),
            n_off=int(self.off_vector.data.value[nbin]),
            a_on=self.on_vector.backscal,
            a_off=self.off_vector.backscal,
            obs_id=self.obs_id,
            livetime=self.livetime,
        )
        return ObservationStats(**kwargs)

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

    # TODO: This should probably go away
    @classmethod
    def from_observation_table(cls, obs_table):
        """Create `~gammapy.spectrum.SpectrumObservationList` from an
        observation table.

        Parameters
        ----------
        obs_table : `~gammapy.data.ObservationTable`
            Observation table with column ``PHAFILE``
        """
        obs = [SpectrumObservation.read(_) for _ in obs_table['PHAFILE']]

        return cls(obs)


# TODO: completely untested?
def stack(cls, obs_list, group_id=None):
    """Stack `~gammapy.spectrum.SpectrumObservation`

    Observation stacking is implemented as follows
    Averaged livetime ratio between ON and OFF regions, arf and rmf
    :math:`\\alpha_{\\mathrm{tot}}`  for all observations is calculated as
    .. math:: \\alpha_{\\mathrm{tot}} = \\frac{\\sum_{i}\\alpha_i \\cdot N_i}{\\sum_{i} N_i}
    .. math:: \\arf_{\\mathrm{tot}} = \\frac{\\sum_{i}\\arf_i \\cdot \\livetime_i}{\\sum_{i} \\livetime_i}
    .. math:: \\rmf_{\\mathrm{tot}} = \\frac{\\sum_{i}\\rmf_i \\cdot arf_i \\cdot livetime_i}{\\sum_{i} arf_i \\cdot livetime_i}

    Parameters
    ----------
    obs_list : list of `~gammapy.spectrum.SpectrumObservations`
        Observations to stack
    group_id : int, optional
        ID for stacked observations

    Returns
    -------
    stacked_obs : `~gammapy.spectrum.SpectrumObservations`
    """

    group_id = obs_list[0].meta.obs_id if group_id is None else group_id

    # Stack ON and OFF vector using the _add__ method in the CountSpectrum class
    on_vec = np.sum([o.on_vector for o in obs_list])

    # If obs_list contains only on element np.sum does not call the
    #  _add__ method which lead to a faulty meta object
    if len(obs_list) == 1:
        on_vec.meta = Bunch(livetime=obs_list[0].meta.livetime,
                            backscal=1)

    on_vec.meta.update(obs_id=group_id)

    off_vec = np.sum([o.off_vector for o in obs_list])

    # Stack arf vector
    arf_band = [o.effective_area.data * o.meta.livetime.value for o in obs_list]
    arf_band_tot = np.sum(arf_band, axis=0)
    livetime_tot = np.sum([o.meta.livetime.value for o in obs_list])
    arf_vec = arf_band_tot / livetime_tot
    energy = obs_list[1].effective_area.energy.data
    data = arf_vec * obs_list[0].effective_area.data.unit

    arf = EffectiveAreaTable(energy=energy, data=data)

    # Stack rmf vector
    rmf_band = [o.energy_dispersion.pdf_matrix.T * o.effective_area.data.value * o.meta.livetime.value for
                o in obs_list]
    rmf_band_tot = np.sum(rmf_band, axis=0)
    pdf_mat = rmf_band_tot / arf_band_tot
    etrue = obs_list[0].energy_dispersion.true_energy
    ereco = obs_list[0].energy_dispersion.reco_energy
    inan = np.isnan(pdf_mat)
    pdf_mat[inan] = 0
    rmf = EnergyDispersion(pdf_mat.T, etrue, ereco)

    # Calculate average alpha
    alpha_band = [o.alpha * o.off_vector.total_counts for o in obs_list]
    alpha_band_tot = np.sum(alpha_band)
    off_tot = np.sum([o.off_vector.total_counts for o in obs_list])
    alpha_mean = alpha_band_tot / off_tot
    off_vec.meta.backscal = 1. / alpha_mean

    # Calculate energy range
    # TODO: for the moment we take the minimal safe energy range
    # Taking the whole range requires an energy dependent lifetime
    emin = max([_.meta.safe_energy_range[0] for _ in obs_list])
    emax = min([_.meta.safe_energy_range[1] for _ in obs_list])

    m = Bunch()
    m['energy_range'] = EnergyBounds([emin, emax])
    on_vec.meta['safe_energy_range'] = EnergyBounds([emin, emax])
    # m['safe_energy_range'] = EnergyBounds([emin, emax])
    m['obs_ids'] = [o.meta.obs_id for o in obs_list]
    m['alpha_method1'] = alpha_mean
    m['livetime'] = Quantity(livetime_tot, "s")
    m['group_id'] = group_id

    return cls(on_vec, off_vec, rmf, arf, meta=m)
