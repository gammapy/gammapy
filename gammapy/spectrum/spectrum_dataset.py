# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import numpy as np
from astropy.units import Quantity
from ..utils.scripts import make_path
from ..irf import EffectiveAreaTable, EnergyDispersion
from .core import PHACountsSpectrum
from .utils import SpectrumEvaluator
from ..utils.fitting import Dataset

__all__ = [
    "ONOFFSpectrumDataset"
]


class ONOFFSpectrumDataset(Dataset):
    """Compute spectral model fit statistic on a ON OFF Spectrum.


    Parameters
    ----------
    model : `~gammapy.spectrum.models.SpectralModel`
        Fit model
    ONcounts : `~gammapy.spectrum.PHACountsSpectrum`
        ON Counts spectrum
    OFFcounts : `~gammapy.spectrum.PHACountsSpectrum`
        ON Counts spectrum
    livetime : float
        Livetime
    mask : numpy.array
        Mask to apply to the likelihood.
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective area
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    """

    def __init__(
            self,
            model=None,
            ONcounts=None,
            OFFcounts=None,
            livetime=None,
            mask=None,
            aeff=None,
            edisp=None,
    ):
        if mask is not None and mask.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.model = model
        self.ONcounts = ONcounts
        self.OFFcounts = OFFcounts
        self.livetime = livetime
        self.mask = mask
        self.aeff = aeff
        self.edisp = edisp

    @property
    def alpha(self):
        return self.ONcounts.backscal / self.OFFcounts.backscal

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        if model is not None:
            self._parameters = Parameters(self._model.parameters.parameters)
            if self.edisp is None:
                self._predictor = SpectrumEvaluator(model=self.model,
                                                    livetime=self.livetime,
                                                    aeff=self.aeff,
                                                    e_true=self.ONcounts.energy.bins)
            else:
                self._predictor = SpectrumEvaluator(model=self.model,
                                                    aeff=self.aeff,
                                                    edisp=self.edisp,
                                                    livetime=self.livetime)

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
        """Returns npred counts vector """
        if self._predictor is None:
            raise AttributeError("No model set for this Dataset")
        model_npred = self._predictor.compute_npred().data.data
        return model_npred

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        on_stat_ = stats.wstat(
            n_on=self.ONcounts.data.data,
            n_off=self.OFFcounts.data.data,
            alpha=self.alpha,
            mu_sig=self.npred(),
        )
        return np.nan_to_num(on_stat_)

    def likelihood(self, parameters, mask=None):
        """Total likelihood given the current model parameters.

        Parameters
        ----------
        mask : `~numpy.ndarray`
            Mask to be combined with the dataset mask.
        """
        if self.mask is None and mask is None:
            stat = self.likelihood_per_bin()
        elif self.mask is None:
            stat = self.likelihood_per_bin()[mask]
        elif mask is None:
            stat = self.likelihood_per_bin()[self.mask]
        else:
            stat = self.likelihood_per_bin()[mask & self.mask]

        return np.sum(stat, dtype=np.float64)

    @classmethod
    def read_from_ogip(cls, filename):
        """Read from OGIP files.

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

        try:
            effective_area = EffectiveAreaTable.read(str(dirname / arf))
        except IOError:
            # TODO : Add logger and echo warning
            effective_area = None

        return cls(
            ONcounts=on_vector,
            aeff=effective_area,
            OFFcounts=off_vector,
            edisp=energy_dispersion,
            livetime=on_vector.livetime
        )

    def export_to_ogip(self, outdir=None, overwrite=False):
        """Write OGIP files.

        Parameters
        ----------
        outdir : `pathlib.Path`
            output directory, default: pwd
        overwrite : bool
            Overwrite existing files?
        """
        outdir = Path.cwd() if outdir is None else Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        phafile = self.ONcounts.phafile
        bkgfile = self.ONcounts.bkgfile
        arffile = self.ONcounts.arffile
        rmffile = self.ONcounts.rmffile

        self.ONcounts.write(outdir / phafile, overwrite=overwrite)
        self.aeff.write(outdir / arffile, overwrite=overwrite)
        if self.ONcounts is not None:
            self.ONcounts.write(outdir / bkgfile, overwrite=overwrite)
        if self.edisp is not None:
            self.edisp.write(str(outdir / rmffile), overwrite=overwrite)


