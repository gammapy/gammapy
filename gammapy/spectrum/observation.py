# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from copy import deepcopy
from astropy.units import Quantity
from ..extern.six.moves import UserList  # pylint:disable=import-error
from ..extern.pathlib import Path
from ..utils.scripts import make_path
from ..utils.energy import EnergyBounds
from ..utils.table import table_from_row_data
from ..data import ObservationStats
from ..irf import EffectiveAreaTable, EnergyDispersion
from ..irf import IRFStacker
from .core import CountsSpectrum, PHACountsSpectrum, PHACountsSpectrumList
from .utils import CountsPredictor

__all__ = [
    "SpectrumStats",
    "SpectrumObservation",
    "SpectrumObservationList",
    "SpectrumObservationStacker",
]


class SpectrumStats(ObservationStats):
    """Spectrum stats.

    Extends `~gammapy.data.ObservationStats` with spectrum
    specific information (energy bin info at the moment).
    """

    def __init__(self, **kwargs):
        self.energy_min = kwargs.pop("energy_min", Quantity(0, "TeV"))
        self.energy_max = kwargs.pop("energy_max", Quantity(0, "TeV"))
        super(SpectrumStats, self).__init__(**kwargs)

    def __str__(self):
        ss = super(SpectrumStats, self).__str__()
        ss += "energy range: {:.2f} - {:.2f}".format(self.energy_min, self.energy_max)
        return ss

    def to_dict(self):
        """TODO: document"""
        data = super(SpectrumStats, self).to_dict()
        data["energy_min"] = self.energy_min
        data["energy_max"] = self.energy_max
        return data


class SpectrumObservation(object):
    """1D spectral analysis storage class.

    This container holds the ingredients for 1D region based spectral analysis.

    Meta data is stored in the ``on_vector`` attribute.
    This reflects the OGIP convention.

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
    ::

        from gammapy.spectrum import SpectrumObservation
        filename = '$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits'
        obs = SpectrumObservation.read(filename)
        print(obs)
    """

    def __init__(self, on_vector, aeff=None, off_vector=None, edisp=None):
        self.on_vector = on_vector
        self.aeff = aeff
        self.off_vector = off_vector
        self.edisp = edisp

    def __str__(self):
        ss = self.total_stats_safe_range.__str__()
        return ss

    @property
    def obs_id(self):
        """Unique identifier"""
        return self.on_vector.obs_id

    @obs_id.setter
    def obs_id(self, obs_id):
        self.on_vector.obs_id = obs_id
        if self.off_vector is not None:
            self.off_vector.obs_id = obs_id

    @property
    def meta(self):
        """Meta information"""
        return self.on_vector.meta

    @property
    def livetime(self):
        """Dead-time corrected observation time"""
        return self.on_vector.livetime

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions"""
        return self.on_vector.backscal / self.off_vector.backscal

    @property
    def e_reco(self):
        """Reconstruced energy bounds array."""
        return EnergyBounds(self.on_vector.energy.bins)

    @property
    def e_true(self):
        """True energy bounds array."""
        return EnergyBounds(self.aeff.energy.bins)

    @property
    def nbins(self):
        """Number of reconstruced energy bins"""
        return self.on_vector.energy.nbins

    @property
    def lo_threshold(self):
        """Low energy threshold"""
        return self.on_vector.lo_threshold

    @lo_threshold.setter
    def lo_threshold(self, threshold):
        self.on_vector.lo_threshold = threshold
        if self.off_vector is not None:
            self.off_vector.lo_threshold = threshold

    @property
    def hi_threshold(self):
        """High energy threshold"""
        return self.on_vector.hi_threshold

    @hi_threshold.setter
    def hi_threshold(self, threshold):
        self.on_vector.hi_threshold = threshold
        if self.off_vector is not None:
            self.off_vector.hi_threshold = threshold

    def reset_thresholds(self):
        """Reset energy thresholds (i.e. declare all energy bins valid)"""
        self.on_vector.reset_thresholds()
        if self.off_vector is not None:
            self.off_vector.reset_thresholds()

    def compute_energy_threshold(
        self, method_lo="none", method_hi="none", reset=False, **kwargs
    ):
        """Compute and set the safe energy threshold.

        Set the high and low energy threshold for each observation based on a
        chosen method.

        Available methods for setting the low energy threshold:

        * area_max : Set energy threshold at x percent of the maximum effective
          area (x given as kwargs['area_percent_lo'])

        * energy_bias : Set energy threshold at energy where the energy bias
          exceeds a value of x percent (given as kwargs['bias_percent_lo'])

        * none : Do not apply a lower threshold

        Available methods for setting the high energy threshold:

        * area_max : Set energy threshold at x percent of the maximum effective
          area (x given as kwargs['area_percent_hi'])

        * energy_bias : Set energy threshold at energy where the energy bias
          exceeds a value of x percent (given as kwargs['bias_percent_hi'])

        * none : Do not apply a higher energy threshold

        Parameters
        ----------
        method_lo : {'area_max', 'energy_bias', 'none'}
            Method for defining the low energy threshold

        method_hi : {'area_max', 'energy_bias', 'none'}
            Method for defining the high energy threshold

        reset : bool
            Reset existing energy thresholds before setting the new ones
            (default is `False`)
        """

        if reset:
            self.reset_thresholds()

        # It is important to update the low and high threshold for ON and OFF
        # vector, otherwise Sherpa will not understand the files

        # Low threshold
        if method_lo == "area_max":
            aeff_thres = kwargs["area_percent_lo"] / 100 * self.aeff.max_area
            thres_lo = self.aeff.find_energy(aeff_thres)
        elif method_lo == "energy_bias":
            thres_lo = self._find_bias_energy(kwargs["bias_percent_lo"] / 100)
        elif method_lo == "none":
            thres_lo = self.e_true[0]
        else:
            raise ValueError("Undefine method for low threshold: {}".format(method_lo))

        self.on_vector.lo_threshold = thres_lo
        if self.off_vector is not None:
            self.off_vector.lo_threshold = thres_lo

        # High threshold
        if method_hi == "area_max":
            aeff_thres = kwargs["area_percent_hi"] / 100 * self.aeff.max_area
            thres_hi = self.aeff.find_energy(aeff_thres, reverse=True)
        elif method_hi == "energy_bias":
            thres_hi = self._find_bias_energy(
                kwargs["bias_percent_hi"] / 100, reverse=True
            )
        elif method_hi == "none":
            thres_hi = self.e_true[-1]
        else:
            raise ValueError(
                "Undefined method for high threshold: {}".format(method_hi)
            )

        self.on_vector.hi_threshold = thres_hi
        if self.off_vector is not None:
            self.off_vector.hi_threshold = thres_hi

    def _find_bias_energy(self, bias_value, reverse=False):
        """Helper function to interpolate between bias values to retrieve an energy"""
        e = self.e_true.log_centers
        bias = np.abs(self.edisp.get_bias(e))
        with np.errstate(invalid="ignore"):
            valid = np.where(bias <= bias_value)[0]
        idx = valid[0]
        if reverse:
            idx = valid[-1]
        if not reverse:
            if idx == 0:
                energy = self.e_true[idx].value
            else:
                energy = np.interp(
                    bias_value, (bias[[idx - 1, idx]].value), (e[[idx - 1, idx]].value)
                )
        else:
            if idx == e.size - 1:
                energy = self.e_true[idx + 1].value
            else:
                energy = np.interp(
                    bias_value, (bias[[idx, idx + 1]].value), (e[[idx, idx + 1]].value)
                )
        return energy * self.e_true.unit

    @property
    def background_vector(self):
        """Background `~gammapy.spectrum.CountsSpectrum`.

        bkg = alpha * n_off

        If alpha is a function of energy this will differ from
        self.on_vector * self.total_stats.alpha because the latter returns an
        average value for alpha.
        """
        energy = self.off_vector.energy
        data = self.off_vector.data.data * self.alpha
        return CountsSpectrum(data=data, energy_lo=energy.lo, energy_hi=energy.hi)

    @property
    def excess_vector(self):
        """Excess `~gammapy.spectrum.CountsSpectrum`.

        excess = n_on = alpha * n_off
        """
        energy = self.off_vector.energy
        data = self.on_vector.data.data - self.background_vector.data.data
        return CountsSpectrum(data=data, energy_lo=energy.lo, energy_hi=energy.hi)

    @property
    def total_stats(self):
        """Return total `~gammapy.spectrum.SpectrumStats`
        """
        return self.stats_in_range(0, self.nbins - 1)

    @property
    def total_stats_safe_range(self):
        """Return total `~gammapy.spectrum.SpectrumStats` within the tresholds
        """
        safe_bins = self.on_vector.bins_in_safe_range
        return self.stats_in_range(safe_bins[0], safe_bins[-1])

    def stats_in_range(self, bin_min, bin_max):
        """Compute stats for a range of energy bins.

        Parameters
        ----------
        bin_min, bin_max: int
            Bins to include

        Returns
        -------
        stats : `~gammapy.spectrum.SpectrumStats`
            Stacked stats
        """
        idx = np.arange(bin_min, bin_max + 1)
        stats_list = [self.stats(ii) for ii in idx]
        stacked_stats = SpectrumStats.stack(stats_list)
        stacked_stats.livetime = self.livetime
        stacked_stats.gamma_rate = stacked_stats.excess / stacked_stats.livetime
        stacked_stats.obs_id = self.obs_id
        stacked_stats.energy_min = self.e_reco[bin_min]
        stacked_stats.energy_max = self.e_reco[bin_max + 1]
        return stacked_stats

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
        if self.off_vector is not None:
            n_off = int(self.off_vector.data.data.value[idx])
            a_off = self.off_vector._backscal_array[idx]
        else:
            n_off = 0
            a_off = 1  # avoid zero division error

        return SpectrumStats(
            energy_min=self.e_reco[idx],
            energy_max=self.e_reco[idx + 1],
            n_on=int(self.on_vector.data.data.value[idx]),
            n_off=n_off,
            a_on=self.on_vector._backscal_array[idx],
            a_off=a_off,
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
        rows = [self.stats(idx).to_dict() for idx in range(len(self.e_reco) - 1)]
        return table_from_row_data(rows=rows)

    def predicted_counts(self, model):
        """Calculated number of predicted counts given a model.

        Parameters
        ----------
        model : `~gammapy.spectrum.models.SpectralModel`
            Spectral model

        Returns
        -------
        npred : `~gammapy.spectrum.CountsSpectrum`
            Predicted counts
        """
        predictor = CountsPredictor(
            model=model, edisp=self.edisp, aeff=self.aeff, livetime=self.livetime
        )
        predictor.run()
        return predictor.npred

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

        return cls(
            on_vector=on_vector,
            aeff=effective_area,
            off_vector=off_vector,
            edisp=energy_dispersion,
        )

    def write(self, outdir=None, use_sherpa=False, overwrite=False):
        """Write OGIP files.

        If you want to use the written files with Sherpa you have to set the
        ``use_sherpa`` flag. Then all files will be written in units 'keV' and
        'cm2'.

        Parameters
        ----------
        outdir : `~gammapy.extern.pathlib.Path`
            output directory, default: pwd
        use_sherpa : bool, optional
            Write Sherpa compliant files, default: False
        overwrite : bool
            Overwrite existing files?
        """
        outdir = Path.cwd() if outdir is None else Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        phafile = self.on_vector.phafile
        bkgfile = self.on_vector.bkgfile
        arffile = self.on_vector.arffile
        rmffile = self.on_vector.rmffile

        # Write in keV and cm2 for sherpa
        if use_sherpa:
            # TODO: Change this implementation.
            # write should not change the object
            # put this code in a separate method that makes a copy with the changes.
            # then call `.write` on that here, or remove the option and let the user do it.
            self.on_vector.energy.lo = self.on_vector.energy.lo.to("keV")
            self.on_vector.energy.hi = self.on_vector.energy.hi.to("keV")
            self.aeff.energy.lo = self.aeff.energy.lo.to("keV")
            self.aeff.energy.hi = self.aeff.energy.hi.to("keV")
            self.aeff.data.data = self.aeff.data.data.to("cm2")
            if self.off_vector is not None:
                self.off_vector.energy.lo = self.off_vector.energy.lo.to("keV")
                self.off_vector.energy.hi = self.off_vector.energy.hi.to("keV")
            if self.edisp is not None:
                self.edisp.e_reco.lo = self.edisp.e_reco.lo.to("keV")
                self.edisp.e_reco.hi = self.edisp.e_reco.hi.to("keV")
                self.edisp.e_true.lo = self.edisp.e_true.lo.to("keV")
                self.edisp.e_true.hi = self.edisp.e_true.hi.to("keV")
                # Set data to itself to trigger reset of the interpolator
                # TODO: Make NDData notice change of axis
                self.edisp.data.data = self.edisp.data.data

        self.on_vector.write(outdir / phafile, overwrite=overwrite)
        self.aeff.write(outdir / arffile, overwrite=overwrite)
        if self.off_vector is not None:
            self.off_vector.write(outdir / bkgfile, overwrite=overwrite)
        if self.edisp is not None:
            self.edisp.write(str(outdir / rmffile), overwrite=overwrite)

    def peek(self, figsize=(10, 10)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        ax1.set_title("Counts")
        energy_unit = "TeV"
        if self.off_vector is not None:
            self.background_vector.plot_hist(
                ax=ax1, label="alpha * n_off", color="darkblue", energy_unit=energy_unit
            )
        self.on_vector.plot_hist(
            ax=ax1,
            label="n_on",
            color="darkred",
            energy_unit=energy_unit,
            show_energy=(self.hi_threshold, self.lo_threshold),
        )
        ax1.set_xlim(
            0.7 * self.lo_threshold.to(energy_unit).value,
            1.3 * self.hi_threshold.to(energy_unit).value,
        )
        ax1.legend(numpoints=1)

        ax2.set_title("Effective Area")
        e_unit = self.aeff.energy.unit
        self.aeff.plot(ax=ax2, show_energy=(self.hi_threshold, self.lo_threshold))
        ax2.set_xlim(
            0.7 * self.lo_threshold.to(e_unit).value,
            1.3 * self.hi_threshold.to(e_unit).value,
        )

        ax3.axis("off")
        if self.off_vector is not None:
            ax3.text(0, 0.2, "{}".format(self.total_stats_safe_range), fontsize=12)

        ax4.set_title("Energy Dispersion")
        if self.edisp is not None:
            self.edisp.plot_matrix(ax=ax4)

        # TODO: optimize layout
        plt.subplots_adjust(wspace=0.3)

    def to_sherpa(self):
        """Convert to `~sherpa.astro.data.DataPHA`.

        Associated background vectors and IRFs are also translated to sherpa
        objects and appended to the PHA instance.
        """
        pha = self.on_vector.to_sherpa(name="pha_obs{}".format(self.obs_id))
        if self.aeff is not None:
            arf = self.aeff.to_sherpa(name="arf_obs{}".format(self.obs_id))
        else:
            arf = None
        if self.edisp is not None:
            rmf = self.edisp.to_sherpa(name="rmf_obs{}".format(self.obs_id))
        else:
            rmf = None

        pha.set_response(arf, rmf)

        if self.off_vector is not None:
            bkg = self.off_vector.to_sherpa(name="bkg_obs{}".format(self.obs_id))
            bkg.set_response(arf, rmf)
            pha.set_background(bkg, 1)

        # see https://github.com/sherpa/sherpa/blob/36c1f9dabb3350b64d6f54ab627f15c862ee4280/sherpa/astro/data.py#L1400
        pha._set_initial_quantity()
        return pha

    def copy(self):
        """A deep copy."""
        return deepcopy(self)


class SpectrumObservationList(UserList):
    """List of `~gammapy.spectrum.SpectrumObservation` objects."""

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\nNumber of observations: {}".format(len(self))
        # ss += '\n{}'.format(self.obs_id)
        return ss

    def obs(self, obs_id):
        """Return one observation.

        Parameters
        ----------
        obs_id : int
            Identifier
        """
        obs_id_list = [o.obs_id for o in self]
        idx = obs_id_list.index(obs_id)
        return self[idx]

    @property
    def obs_id(self):
        """List of observations ids"""
        return [o.obs_id for o in self]

    @property
    def total_livetime(self):
        """Summed livetime"""
        livetimes = [o.livetime.to("s").value for o in self]
        return Quantity(np.sum(livetimes), "s")

    @property
    def on_vector_list(self):
        """On `~gammapy.spectrum.PHACountsSpectrumList`"""
        return PHACountsSpectrumList([o.on_vector for o in self])

    @property
    def off_vector_list(self):
        """Off `~gammapy.spectrum.PHACountsSpectrumList`"""
        return PHACountsSpectrumList([o.off_vector for o in self])

    def stack(self):
        """Return stacked `~gammapy.spectrum.SpectrumObservation`"""
        stacker = SpectrumObservationStacker(obs_list=self)
        stacker.run()
        return stacker.stacked_obs

    def safe_range(self, method="inclusive"):
        """Safe energy range

        This is the energy range in with any / all observations have their safe
        threshold

        Parameters
        ----------
        method : str, {'inclusive', 'exclusive'}
            Maximum or minimum range
        """
        unit = "TeV"
        lo = [obs.lo_threshold.to(unit).value for obs in self]
        hi = [obs.hi_threshold.to(unit).value for obs in self]

        if method == "inclusive":
            return Quantity([min(lo), max(hi)], unit)
        elif method == "exclusive":
            return Quantity([max(lo), min(hi)], unit)
        else:
            raise ValueError("Invalid method: {}".format(method))

    def write(self, outdir=None, pha_typeII=False, **kwargs):
        """Create OGIP files

        Each observation will be written as seperate set of FITS files by
        default. If the option ``pha_typeII`` is enabled all on and off counts
        spectra will be collected into one
        `~gammapy.spectrum.PHACountsSpectrumList` and written to one FITS file.
        All datasets will be associated to the same response files.
        see
        https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node8.html

        TODO: File written with the ``pha_typeII`` option are not read
        properly with sherpa. This could be a sherpa issue. Investigate and
        file issue.

        Parameters
        ----------
        outdir : str, `~gammapy.extern.pathlib.Path`, optional
            Output directory, default: pwd
        pha_typeII : bool, default: False
            Collect PHA datasets into one file
        """
        outdir = make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        if not pha_typeII:
            for obs in self:
                obs.write(outdir=outdir, **kwargs)
        else:
            onlist = self.on_vector_list
            onlist.write(outdir / "pha2.fits", **kwargs)
            offlist = self.off_vector_list
            # This filename is hardcoded since it is a column in the on list
            offlist.write(outdir / "bkg.fits", **kwargs)
            arf_file = onlist.to_table().meta["ancrfile"]
            rmf_file = onlist.to_table().meta["respfile"]
            self[0].aeff.write(outdir / arf_file, **kwargs)
            self[0].edisp.write(outdir / rmf_file, **kwargs)

    @classmethod
    def read(cls, directory, pha_typeII=False):
        """Read multiple observations

        This methods reads all PHA files contained in a given directory. Enable
        ``pha_typeII`` to read a PHA type II file.

        see
        https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node8.html

        TODO: Replace with more sophisticated file managment system

        Parameters
        ----------
        directory : `~gammapy.extern.pathlib.Path`
            Directory holding the observations
        pha_typeII : bool, default: False
            Read PHA typeII file
        """
        obs_list = cls()
        directory = make_path(directory)

        if not pha_typeII:
            # glob default order depends on OS, so we call sorted() explicitely to
            # get reproducable results
            filelist = sorted(directory.glob("pha*.fits"))
            for phafile in filelist:
                obs = SpectrumObservation.read(phafile)
                obs_list.append(obs)
        else:
            # NOTE: filenames for type II PHA files are hardcoded
            on_vectors = PHACountsSpectrumList.read(directory / "pha2.fits")
            off_vectors = PHACountsSpectrumList.read(directory / "bkg.fits")
            aeff = EffectiveAreaTable.read(directory / "arf.fits")
            edisp = EnergyDispersion.read(directory / "rmf.fits")

            for on, off in zip(on_vectors, off_vectors):
                obs = SpectrumObservation(
                    on_vector=on, off_vector=off, aeff=aeff, edisp=edisp
                )
                obs_list.append(obs)

        return obs_list

    def peek(self):
        """Quickly look at observations

        Uses IPython widgets.
        TODO: Change to bokeh
        """
        from ipywidgets import interact

        max_ = len(self) - 1

        def show_obs(idx):
            self[idx].peek()

        return interact(show_obs, idx=(0, max_, 1))


class SpectrumObservationStacker(object):
    r"""Stack observations in a `~gammapy.spectrum.SpectrumObservationList`.

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

        \overline{\alpha}_k =
        \frac{\overline{{b_{on}}}_k}{\overline{{b_{off}}}_k}

        \overline{{b}_{on}}_k = 1

        \overline{{b}_{off}}_k = \frac{1}{\sum_{j}\alpha_{jk} \cdot
            \mathrm{n_{off}}_{jk} \cdot \epsilon_{jk}} \cdot \overline{\mathrm {n_{off}}}

    Please refer to the `~gammapy.irf.IRFStacker` for the description
    of how the IRFs are stacked.

    Parameters
    ----------
    obs_list : `~gammapy.spectrum.SpectrumObservationList`
        Observations to stack

    Examples
    --------
    >>> from gammapy.spectrum import SpectrumObservationList, SpectrumObservationStacker
    >>> obs_list = SpectrumObservationList.read('$GAMMAPY_DATA/joint-crab/spectra/hess')
    >>> obs_stacker = SpectrumObservationStacker(obs_list)
    >>> obs_stacker.run()
    >>> print(obs_stacker.stacked_obs)
    *** Observation summary report ***
    Observation Id: [23523-23592]
    Livetime: 0.879 h
    On events: 279
    Off events: 108
    Alpha: 0.037
    Bkg events in On region: 3.96
    Excess: 275.04
    Excess / Background: 69.40
    Gamma rate: 0.14 1 / min
    Bkg rate: 0.00 1 / min
    Sigma: 37.60
    energy range: 681292069.06 keV - 87992254356.91 keV
    """

    def __init__(self, obs_list):
        self.obs_list = SpectrumObservationList(obs_list)
        self.stacked_on_vector = None
        self.stacked_off_vector = None
        self.stacked_aeff = None
        self.stacked_edisp = None
        self.stacked_bkscal_on = None
        self.stacked_bkscal_off = None
        self.stacked_obs = None

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n{}".format(self.obs_list)
        return ss

    def run(self):
        """Run all steps in the correct order."""
        self.stack_counts_vectors()
        self.stack_aeff()
        self.stack_edisp()
        self.stack_obs()

    def stack_counts_vectors(self):
        """Stack on and off vectors."""
        self.stack_on_vector()
        self.stack_off_vector()
        self.stack_backscal()
        self.setup_counts_vectors()

    def stack_on_vector(self):
        """Stack the on count vector."""
        on_vector_list = [o.on_vector for o in self.obs_list]
        self.stacked_on_vector = self.stack_counts_spectrum(on_vector_list)

    def stack_off_vector(self):
        """Stack the off count vector."""
        off_vector_list = [o.off_vector for o in self.obs_list]
        self.stacked_off_vector = self.stack_counts_spectrum(off_vector_list)

    @staticmethod
    def stack_counts_spectrum(counts_spectrum_list):
        """Stack `~gammapy.spectrum.PHACountsSpectrum`.

        * Bins outside the safe energy range are set to 0
        * Attributes are set to None.
        * The quality vector of the observations are combined with a logical or,
          such that the low (high) threshold of the stacked obs is the minimum
          low (maximum high) threshold of the observation list to be stacked.
        """
        template = counts_spectrum_list[0].copy()
        energy = template.energy
        stacked_data = np.zeros(energy.nbins)
        stacked_quality = np.ones(energy.nbins)
        for spec in counts_spectrum_list:
            stacked_data += spec.counts_in_safe_range.value
            temp = np.logical_and(stacked_quality, spec.quality)
            stacked_quality = np.array(temp, dtype=int)

        return PHACountsSpectrum(
            data=stacked_data,
            energy_lo=energy.lo,
            energy_hi=energy.hi,
            quality=stacked_quality,
        )

    def stack_backscal(self):
        """Stack ``backscal`` for on and off vector."""
        nbins = self.obs_list[0].e_reco.nbins
        bkscal_on = np.ones(nbins)
        bkscal_off = np.zeros(nbins)

        alpha_sum = 0.0

        for obs in self.obs_list:
            bkscal_on_data = obs.on_vector._backscal_array.copy()
            bkscal_off_data = obs.off_vector._backscal_array.copy()
            bkscal_off += (
                bkscal_on_data / bkscal_off_data
            ) * obs.off_vector.counts_in_safe_range.value
            alpha_sum += (obs.alpha * obs.off_vector.counts_in_safe_range).sum()

        with np.errstate(divide="ignore", invalid="ignore"):
            stacked_bkscal_off = self.stacked_off_vector.data.data.value / bkscal_off
            alpha_average = (
                alpha_sum / self.stacked_off_vector.counts_in_safe_range.sum()
            )

        # there should be no nan values in backscal_on or backscal_off
        # this leads to problems when fitting the data
        # use 1 for backscale of on_vector and 1 / alpha_average for backscale of off_vector
        alpha_correction = 1
        idx = np.where(self.stacked_off_vector.data.data == 0)[0]
        bkscal_on[idx] = alpha_correction
        # For the bins where the stacked OFF counts equal 0, the alpha value is performed by weighting on the total
        # OFF counts of each run
        stacked_bkscal_off[idx] = alpha_correction / alpha_average

        self.stacked_bkscal_on = bkscal_on
        self.stacked_bkscal_off = stacked_bkscal_off

    def setup_counts_vectors(self):
        """Add correct attributes to stacked counts vectors."""
        total_livetime = self.obs_list.total_livetime
        self.stacked_on_vector.livetime = total_livetime
        self.stacked_off_vector.livetime = total_livetime
        self.stacked_on_vector.backscal = self.stacked_bkscal_on
        self.stacked_off_vector.backscal = self.stacked_bkscal_off
        self.stacked_on_vector.obs_id = self.obs_list.obs_id
        self.stacked_off_vector.obs_id = self.obs_list.obs_id

    def stack_aeff(self):
        """Stack effective areas (weighted by livetime).

        Calls `gammapy.irf.IRFStacker.stack_aeff`.
        """
        irf_stacker = IRFStacker(
            list_aeff=[obs.aeff for obs in self.obs_list],
            list_livetime=[obs.livetime for obs in self.obs_list],
        )
        irf_stacker.stack_aeff()
        self.stacked_aeff = irf_stacker.stacked_aeff

    def stack_edisp(self):
        """Stack energy dispersion (weighted by exposure).

        Calls `~gammapy.irf.IRFStacker.stack_edisp`
        """
        irf_stacker = IRFStacker(
            list_aeff=[obs.aeff for obs in self.obs_list],
            list_livetime=[obs.livetime for obs in self.obs_list],
            list_edisp=[obs.edisp for obs in self.obs_list],
            list_low_threshold=[obs.lo_threshold for obs in self.obs_list],
            list_high_threshold=[obs.hi_threshold for obs in self.obs_list],
        )
        irf_stacker.stack_edisp()
        self.stacked_edisp = irf_stacker.stacked_edisp

    def stack_obs(self):
        """Create stacked `~gammapy.spectrum.SpectrumObservation`"""
        self.stacked_obs = SpectrumObservation(
            on_vector=self.stacked_on_vector,
            off_vector=self.stacked_off_vector,
            aeff=self.stacked_aeff,
            edisp=self.stacked_edisp,
        )
