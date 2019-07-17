# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import OrderedDict
import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from gammapy.spectrum.core import SpectrumEvaluator
from ..utils.scripts import make_path
from ..utils.fitting import Dataset, Parameters
from ..utils.fits import energy_axis_to_ebounds
from ..stats import wstat, cash
from ..utils.random import get_random_state
from ..data import ObservationStats
from .core import CountsSpectrum
from ..irf import EffectiveAreaTable, EnergyDispersion, IRFStacker


__all__ = ["SpectrumDatasetOnOff", "SpectrumDataset", "SpectrumDatasetOnOffStacker"]


class SpectrumDataset(Dataset):
    """Spectrum dataset for likelihood fitting.

    The spectrum dataset bundles reduced counts data, with a spectral model,
    background model and instrument response function to compute the fit-statistic
    given the current model and data.

    Parameters
    ----------
    model : `~gammapy.spectrum.image.SpectralModel`
        Fit model
    counts : `~gammapy.spectrum.CountsSpectrum`
        Counts spectrum
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    background : `~gammapy.spectrum.CountsSpectrum`
        Background to use for the fit.
    mask_safe : `~numpy.ndarray`
        Mask defining the safe data range.
    mask_fit : `~numpy.ndarray`
        Mask to apply to the likelihood for fitting.

    See Also
    --------
    SpectrumDatasetOnOff, FluxPointsDataset, MapDataset

    """

    def __init__(
        self,
        model=None,
        counts=None,
        livetime=None,
        aeff=None,
        edisp=None,
        background=None,
        mask_safe=None,
        mask_fit=None,
        obs_id=None,
    ):
        if mask_fit is not None and mask_fit.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.counts = counts
        self.livetime = livetime
        self.mask_fit = mask_fit
        self.aeff = aeff
        self.edisp = edisp
        self.background = background
        self.model = model
        self.mask_safe = mask_safe
        self.obs_id = obs_id

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if model is not None:
            self._parameters = Parameters(self._model.parameters.parameters)
            self._predictor = SpectrumEvaluator(
                model=self.model,
                livetime=self.livetime,
                aeff=self.aeff,
                e_true=self.counts.energy.edges,
                edisp=self.edisp
            )
        else:
            self._parameters = None
            self._predictor = None

    @property
    def parameters(self):
        if self._parameters is None:
            raise AttributeError("No model set for Dataset")
        else:
            return self._parameters

    @property
    def data_shape(self):
        """Shape of the counts data"""
        return self.counts.data.shape

    def npred(self):
        """Returns npred map (model + background)"""
        npred = self._predictor.compute_npred()
        if self.background:
            npred.data += self.background.data
        return npred

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    def _as_counts_spectrum(self, data):
        energy = self.counts.energy.edges
        return CountsSpectrum(data=data, energy_lo=energy[:-1], energy_hi=energy[1:])

    @property
    def excess(self):
        """Excess (counts - alpha * counts_off)"""
        excess = self.counts.data - self.background.data
        return self._as_counts_spectrum(excess)

    def residuals(self):
        """Residuals (npred_sig - excess)."""
        residuals = self.npred().data - self.excess.data
        return self._as_counts_spectrum(residuals)

    def fake(self, random_state="random-seed"):
        """Simulate a fake `~gammapy.spectrum.CountsSpectrum`.

        Parameters
        ----------
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
                Defines random number generator initialisation.
                Passed to `~gammapy.utils.random.get_random_state`.

        Returns
        -------
        spectrum : `~gammapy.spectrum.CountsSpectrum`
            the fake count spectrum
        """
        random_state = get_random_state(random_state)
        data = random_state.poisson(self.npred().data)
        energy = self.counts.energy.edges
        return CountsSpectrum(energy[:-1], energy[1:], data)

    @property
    def energy_range(self):
        """Energy range defined by the safe mask"""
        energy = self.counts.energy.edges
        e_lo = energy[:-1][self.mask_safe]
        e_hi = energy[1:][self.mask_safe]
        return u.Quantity([e_lo.min(), e_hi.max()])

    def plot_fit(self):
        """Plot counts and residuals in two panels.

        Calls ``plot_counts`` and ``plot_residuals``.
        """
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt

        gs = GridSpec(7, 1)

        ax_spectrum = plt.subplot(gs[:5, :])
        self.plot_counts(ax=ax_spectrum)

        ax_spectrum.set_xticks([])

        ax_residuals = plt.subplot(gs[5:, :])
        self.plot_residuals(ax=ax_residuals)
        return ax_spectrum, ax_residuals

    @property
    def _e_unit(self):
        return self.counts.energy.unit

    def plot_counts(self, ax=None):
        """Plot predicted and detected counts.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        self.npred().plot(ax=ax, label="mu_src", energy_unit=self._e_unit)
        self.excess.plot(ax=ax, label="Excess", fmt=".", energy_unit=self._e_unit)

        e_min, e_max = self.energy_range
        kwargs = {"color": "black", "linestyle": "dashed"}
        ax.axvline(e_min.to_value(self._e_unit), label="fit range", **kwargs)
        ax.axvline(e_max.to_value(self._e_unit), **kwargs)

        ax.legend(numpoints=1)
        ax.set_title("")
        return ax

    def plot_residuals(self, ax=None):
        """Plot residuals.

        Parameters
        ----------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.

        Returns
        -------
        ax : `~matplotlib.pyplot.Axes`
            Axes object.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        residuals = self.residuals()

        residuals.plot(ax=ax, ecolor="black", fmt="none", energy_unit=self._e_unit)
        ax.axhline(0, color="black", lw=0.5)

        ymax = 1.2 * max(residuals.data)
        ax.set_ylim(-ymax, ymax)

        ax.set_xlabel("Energy [{}]".format(self._e_unit))
        ax.set_ylabel("Residuals")
        return ax


class SpectrumDatasetOnOff(SpectrumDataset):
    """Spectrum dataset for on-off likelihood fitting.

    The on-off spectrum dataset bundles reduced counts data, off counts data,
    with a spectral model, relative background efficiency and instrument
    response functions to compute the fit-statistic given the current model
    and data.

    Parameters
    ----------
    model : `~gammapy.spectrum.image.SpectralModel`
        Fit model
    counts : `~gammapy.spectrum.CountsSpectrum`
        ON Counts spectrum
    counts_off : `~gammapy.spectrum.CountsSpectrum`
        OFF Counts spectrum
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    mask_safe : `~numpy.array`
        Mask defining the safe data range.
    mask_fit : `~numpy.array`
        Mask to apply to the likelihood for fitting.
    backscale : `~numpy.array` or float
        Relative background efficiency in
    obs_id : int or list of int
        Observation id(s) corresponding to the (stacked) dataset.

    See Also
    --------
    SpectrumDataset, FluxPointsDataset, MapDataset

    """

    def __init__(
        self,
        model=None,
        counts=None,
        counts_off=None,
        livetime=None,
        mask_fit=None,
        aeff=None,
        edisp=None,
        mask_safe=None,
        backscale=None,
        backscale_off=None,
        obs_id=None,
    ):

        self.counts = counts
        self.counts_off = counts_off
        self.livetime = livetime
        self.mask_fit = mask_fit
        self.aeff = aeff
        self.edisp = edisp
        self.model = model
        self.mask_safe = mask_safe

        if np.isscalar(backscale):
            backscale = np.ones(counts.energy.nbin) * backscale

        if np.isscalar(backscale_off):
            backscale_off = np.ones(counts.energy.nbin) * backscale_off

        self.backscale = backscale
        self.backscale_off = backscale_off
        self.obs_id = obs_id

    @property
    def background(self):
        """"""
        background = self.alpha * self.counts_off.data
        return self._as_counts_spectrum(background)

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions"""
        return self.backscale / self.backscale_off

    def npred_sig(self):
        """Predicted counts from source model (`CountsSpectrum`)."""
        if self._predictor is None:
            raise AttributeError("No model set for this Dataset")
        npred = self._predictor.compute_npred()
        return npred

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        mu_sig = self.npred_sig().data
        on_stat_ = wstat(
            n_on=self.counts.data,
            n_off=self.counts_off.data,
            alpha=self.alpha,
            mu_sig=mu_sig,
        )
        return np.nan_to_num(on_stat_)

    @classmethod
    def read(cls, filename):
        """Read from file

        For now, filename is assumed to the name of a PHA file where BKG file, ARF, and RMF names
        must be set in the PHA header and be present in the same folder

        Parameters
        ----------
        filename : str
            OGIP PHA file to read
        """
        raise NotImplementedError(
            "To read from an OGIP fits file use SpectrumDatasetOnOff.from_ogip_files."
        )

    def peek(self, figsize=(10, 10)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        e_min, e_max = self.energy_range

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=figsize)

        ax1.set_title("Counts")
        energy_unit = "TeV"

        if self.counts_off is not None:
            self.background.plot_hist(
                ax=ax1, label="alpha * n_off", color="darkblue", energy_unit=energy_unit
            )

        self.counts.plot_hist(
            ax=ax1,
            label="n_on",
            color="darkred",
            energy_unit=energy_unit,
            show_energy=(e_min, e_max),
        )

        ax1.set_xlim(
            0.7 * e_min.to_value(energy_unit), 1.3 * e_max.to_value(energy_unit)
        )
        ax1.legend(numpoints=1)

        ax2.set_title("Effective Area")
        e_unit = self.aeff.energy.unit
        self.aeff.plot(ax=ax2, show_energy=(e_min, e_max))
        ax2.set_xlim(0.7 * e_min.to_value(e_unit), 1.3 * e_max.to_value(e_unit))

        ax3.axis("off")

        if self.counts_off is not None:
            stats = ObservationStats(**self._info_dict(in_safe_energy_range=True))
            ax3.text(0, 0.2, "{}".format(stats), fontsize=12)

        ax4.set_title("Energy Dispersion")
        if self.edisp is not None:
            self.edisp.plot_matrix(ax=ax4)

        # TODO: optimize layout
        plt.subplots_adjust(wspace=0.3)

    def to_ogip_files(self, outdir=None, use_sherpa=False, overwrite=False):
        """Write OGIP files.

        If you want to use the written files with Sherpa you have to set the
        ``use_sherpa`` flag. Then all files will be written in units 'keV' and
        'cm2'.

        Parameters
        ----------
        outdir : `pathlib.Path`
            output directory, default: pwd
        use_sherpa : bool, optional
            Write Sherpa compliant files, default: False
        overwrite : bool
            Overwrite existing files?
        """
        # TODO: refactor and reduce amount of code duplication
        outdir = Path.cwd() if outdir is None else make_path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        if isinstance(self.obs_id, list):
            phafile = "pha_stacked.fits"
        else:
            phafile = "pha_obs{}.fits".format(self.obs_id)

        bkgfile = phafile.replace("pha", "bkg")
        arffile = phafile.replace("pha", "arf")
        rmffile = phafile.replace("pha", "rmf")

        counts_table = self.counts.to_table()
        counts_table["QUALITY"] = np.logical_not(self.mask_safe)
        counts_table["BACKSCAL"] = self.backscale
        counts_table["AREASCAL"] = np.ones(self.backscale.size)
        meta = self._ogip_meta()

        meta["respfile"] = rmffile
        meta["backfile"] = bkgfile
        meta["ancrfile"] = arffile
        meta["hduclas2"] = "TOTAL"
        counts_table.meta = meta

        name = counts_table.meta["name"]
        hdu = fits.BinTableHDU(counts_table, name=name)
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu, self._ebounds_hdu(use_sherpa)])

        hdulist.writeto(str(outdir / phafile), overwrite=overwrite)

        self.aeff.write(outdir / arffile, overwrite=overwrite, use_sherpa=use_sherpa)

        if self.counts_off is not None:
            counts_off_table = self.counts_off.to_table()
            counts_off_table["QUALITY"] = np.logical_not(self.mask_safe)
            counts_off_table["BACKSCAL"] = self.backscale_off
            counts_off_table["AREASCAL"] = np.ones(self.backscale.size)
            meta = self._ogip_meta()
            meta["hduclas2"] = "BKG"

            counts_off_table.meta = meta
            name = counts_off_table.meta["name"]
            hdu = fits.BinTableHDU(counts_off_table, name=name)
            hdulist = fits.HDUList(
                [fits.PrimaryHDU(), hdu, self._ebounds_hdu(use_sherpa)]
            )
            hdulist.writeto(str(outdir / bkgfile), overwrite=overwrite)

        if self.edisp is not None:
            self.edisp.write(
                str(outdir / rmffile), overwrite=overwrite, use_sherpa=use_sherpa
            )

    def _ebounds_hdu(self, use_sherpa):
        energy = self.counts.energy.edges

        if use_sherpa:
            energy = energy.to("keV")

        return energy_axis_to_ebounds(energy)

    def _ogip_meta(self):
        """Meta info for the OGIP data format"""
        meta = OrderedDict()
        meta["name"] = "SPECTRUM"
        meta["hduclass"] = "OGIP"
        meta["hduclas1"] = "SPECTRUM"
        meta["corrscal"] = ""
        meta["chantype"] = "PHA"
        meta["detchans"] = self.counts.energy.nbin
        meta["filter"] = "None"
        meta["corrfile"] = ""
        meta["poisserr"] = True
        meta["hduclas3"] = "COUNT"
        meta["hduclas4"] = "TYPE:1"
        meta["lo_thres"] = self.energy_range[0].to_value("TeV")
        meta["hi_thres"] = self.energy_range[1].to_value("TeV")
        meta["exposure"] = self.livetime.to_value("s")
        meta["obs_id"] = self.obs_id
        return meta

    @classmethod
    def from_ogip_files(cls, filename):
        """Read `~gammapy.spectrum.SpectrumDatasetOnOff` from OGIP files.

        BKG file, ARF, and RMF must be set in the PHA header and be present in
        the same folder.

        Parameters
        ----------
        filename : str
            OGIP PHA file to read
        """
        filename = make_path(filename)
        dirname = filename.parent

        with fits.open(str(filename), memmap=False) as hdulist:
            data = _read_ogip_hdulist(hdulist)

        counts = CountsSpectrum(
            energy_hi=data["energy_hi"], energy_lo=data["energy_lo"], data=data["data"]
        )

        phafile = filename.name

        try:
            rmffile = phafile.replace("pha", "rmf")
            energy_dispersion = EnergyDispersion.read(str(dirname / rmffile))
        except IOError:
            # TODO : Add logger and echo warning
            energy_dispersion = None

        try:
            bkgfile = phafile.replace("pha", "bkg")
            filename = str(dirname / bkgfile)

            with fits.open(str(filename), memmap=False) as hdulist:
                data_bkg = _read_ogip_hdulist(hdulist)
                counts_off = CountsSpectrum(
                    energy_hi=data_bkg["energy_hi"],
                    energy_lo=data_bkg["energy_lo"],
                    data=data_bkg["data"],
                )

                backscale_off = data_bkg["backscal"]
        except IOError:
            # TODO : Add logger and echo warning
            counts_off, backscale_off = None, None

        arffile = phafile.replace("pha", "arf")
        aeff = EffectiveAreaTable.read(str(dirname / arffile))

        mask_safe = np.logical_not(data["quality"])

        return cls(
            counts=counts,
            aeff=aeff,
            counts_off=counts_off,
            edisp=energy_dispersion,
            livetime=data["livetime"],
            mask_safe=mask_safe,
            backscale=data["backscal"],
            backscale_off=backscale_off,
            obs_id=data["obs_id"],
        )

    # TODO: decide on a design for dataset info tables / dicts and make it part
    #  of the public API
    def _info_dict(self, in_safe_energy_range=False):
        """Info dict"""
        info = dict()
        mask = self.mask_safe if in_safe_energy_range else slice(None)

        # TODO: handle energy dependent a_on / a_off
        info["a_on"] = self.backscale[0]
        info["n_on"] = self.counts.data[mask].sum()

        if self.counts_off is not None:
            info["n_off"] = self.counts_off.data[mask].sum()
            info["a_off"] = self.backscale_off[0]
        else:
            info["n_off"] = 0
            info["a_off"] = 1

        info["livetime"] = self.livetime
        info["obs_id"] = self.obs_id
        return info


def _read_ogip_hdulist(hdulist, hdu1="SPECTRUM", hdu2="EBOUNDS"):
    """Create from `~astropy.io.fits.HDUList`."""
    counts_table = Table.read(hdulist[hdu1])
    ebounds = Table.read(hdulist[hdu2])
    emin = ebounds["E_MIN"].quantity
    emax = ebounds["E_MAX"].quantity

    # Check if column are present in the header
    quality = None
    areascal = None
    backscal = None

    if "QUALITY" in counts_table.colnames:
        quality = counts_table["QUALITY"].data
    if "AREASCAL" in counts_table.colnames:
        areascal = counts_table["AREASCAL"].data
    if "BACKSCAL" in counts_table.colnames:
        backscal = counts_table["BACKSCAL"].data

    return dict(
        data=counts_table["COUNTS"],
        backscal=backscal,
        energy_lo=emin,
        energy_hi=emax,
        quality=quality,
        areascal=areascal,
        livetime=counts_table.meta["EXPOSURE"] * u.s,
        obs_id=counts_table.meta["OBS_ID"],
        is_bkg=False,
    )


class SpectrumDatasetOnOffStacker:
    r"""Stack a list of homogeneous datasets.

    The stacking of :math:`j` datasets is implemented as follows.
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
    obs_list : list of `~gammapy.spectrum.SpectrumDatasetOnOff`
        Observations to stack

    Examples
    --------
    >>> from gammapy.spectrum import SpectrumDatasetOnOff, SpectrumDatasetOnOffStacker
    >>> obs_ids = [23523, 23526, 23559, 23592]
    >>> datasets = []
    >>> for obs in obs_ids:
    >>>     filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{}.fits"
    >>>     ds = SpectrumDatasetOnOff.from_ogip_files(filename.format(obs))
    >>>     datasets.append(ds)
    >>> obs_stacker = SpectrumDatasetOnOffStacker(datasets)
    >>> stacked = obs_stacker.run()
    >>> print(stacked.livetime)
    6313.8116406202325 s
    """

    def __init__(self, obs_list):
        self.obs_list = obs_list
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
        return self.stacked_obs

    def stack_counts_vectors(self):
        """Stack on and off vectors."""
        self.stack_on_vector()
        self.stack_off_vector()
        self.stack_backscal()
        self.setup_counts_vectors()

    def stack_on_vector(self):
        """Stack the on count vector."""
        on_vector_list = [o.counts for o in self.obs_list]
        self.stacked_on_vector = self.stack_counts_spectrum(on_vector_list)

    def stack_off_vector(self):
        """Stack the off count vector."""
        off_vector_list = [o.counts_off for o in self.obs_list]
        self.stacked_off_vector = self.stack_counts_spectrum(off_vector_list)

    def stack_counts_spectrum(self, counts_spectrum_list):
        """Stack `~gammapy.spectrum.CountsSpectrum`.

        * Bins outside the safe energy range are set to 0
        * Attributes are set to None.
        * The quality vector of the observations are combined with a logical or,
          such that the low (high) threshold of the stacked obs is the minimum
          low (maximum high) threshold of the observation list to be stacked.
        """
        template = counts_spectrum_list[0].copy()
        energy = template.energy
        stacked_data = np.zeros(energy.nbin)
        stacked_quality = np.ones(energy.nbin)
        for spec, obs in zip(counts_spectrum_list, self.obs_list):
            stacked_data[obs.mask_safe] += spec.data[obs.mask_safe]
            temp = np.logical_and(stacked_quality, ~obs.mask_safe)
            stacked_quality = np.array(temp, dtype=int)

        self.stacked_quality = stacked_quality
        return CountsSpectrum(
            data=stacked_data, energy_lo=energy.edges[:-1], energy_hi=energy.edges[1:]
        )

    def stack_backscal(self):
        """Stack ``backscal`` for on and off vector."""
        nbins = self.obs_list[0].counts.energy.nbin
        bkscal_on = np.ones(nbins)
        bkscal_off = np.zeros(nbins)

        alpha_sum = 0.0

        for obs in self.obs_list:
            bkscal_off[obs.mask_safe] += (obs.alpha * obs.counts_off.data)[
                obs.mask_safe
            ]
            alpha_sum += (obs.alpha * obs.counts_off.data)[obs.mask_safe].sum()

        with np.errstate(divide="ignore", invalid="ignore"):
            stacked_bkscal_off = self.stacked_off_vector.data / bkscal_off
            alpha_average = (
                alpha_sum / self.stacked_off_vector.data[obs.mask_safe].sum()
            )

        # there should be no nan values in backscal_on or backscal_off
        # this leads to problems when fitting the data
        # use 1 for backscale of on_vector and 1 / alpha_average for backscale of off_vector
        alpha_correction = 1
        idx = np.where(self.stacked_off_vector.data == 0)[0]
        bkscal_on[idx] = alpha_correction
        # For the bins where the stacked OFF counts equal 0, the alpha value is performed by weighting on the total
        # OFF counts of each run
        stacked_bkscal_off[idx] = alpha_correction / alpha_average

        self.stacked_bkscal_on = bkscal_on
        self.stacked_bkscal_off = stacked_bkscal_off

    def setup_counts_vectors(self):
        """Add correct attributes to stacked counts vectors."""
        livetimes = [obs.livetime.to_value("s") for obs in self.obs_list]
        self.total_livetime = u.Quantity(np.sum(livetimes), "s")

        self.stacked_on_vector.livetime = self.total_livetime
        self.stacked_off_vector.livetime = self.total_livetime
        self.stacked_on_vector.backscal = self.stacked_bkscal_on
        self.stacked_off_vector.backscal = self.stacked_bkscal_off

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
            list_low_threshold=[obs.energy_range[0] for obs in self.obs_list],
            list_high_threshold=[obs.energy_range[1] for obs in self.obs_list],
        )
        irf_stacker.stack_edisp()
        self.stacked_edisp = irf_stacker.stacked_edisp

    def stack_obs(self):
        """Create stacked `~gammapy.spectrum.SpectrumDatasetOnOff`."""
        self.stacked_obs = SpectrumDatasetOnOff(
            counts=self.stacked_on_vector,
            counts_off=self.stacked_off_vector,
            aeff=self.stacked_aeff,
            edisp=self.stacked_edisp,
            livetime=self.total_livetime,
            mask_safe=np.logical_not(self.stacked_quality),
            backscale=self.stacked_on_vector.backscal,
            backscale_off=self.stacked_off_vector.backscal,
            obs_id=[obs.obs_id for obs in self.obs_list],
        )
