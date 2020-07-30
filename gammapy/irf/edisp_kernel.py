# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from gammapy.maps import MapAxis
from gammapy.maps.utils import edges_from_lo_hi
from gammapy.utils.nddata import NDDataArray
from gammapy.utils.scripts import make_path

__all__ = ["EDispKernel"]


class EDispKernel:
    """Energy dispersion matrix.

    Data format specification: :ref:`gadf:ogip-rmf`

    Parameters
    ----------
    e_true : `~gammapy.maps.MapAxis`
        True energy axis. Its name must be "energy_true"
    e_reco : `~gammapy.maps.MapAxis`
        Reconstructed energy axis. Its name must be "energy"
    data : array_like
        2-dim energy dispersion matrix

    Examples
    --------
    Create a Gaussian energy dispersion matrix::

        from gammapy.maps import MapAxis
        from gammapy.irf import EDispKernel
        energy = MapAxis.from_energy_bounds(0.1,10,10, unit='TeV')
        energy_true = MapAxis.from_energy_bounds(0.1,10,10, unit='TeV', name='energy_true')
        edisp = EDispKernel.from_gauss(
            e_true=energy, e_reco=energy,
            sigma=0.1, bias=0,
        )

    Have a quick look:

    >>> print(edisp)
    >>> edisp.peek()

    """

    default_interp_kwargs = dict(bounds_error=False, fill_value=0, method="nearest")
    """Default Interpolation kwargs for `~NDDataArray`. Fill zeros and do not
    interpolate"""

    def __init__(
        self, e_true, e_reco, data, interp_kwargs=None, meta=None,
    ):
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs

        self.data = NDDataArray(
            axes=[e_true, e_reco], data=data, interp_kwargs=interp_kwargs
        )
        self.meta = meta or {}

    def __str__(self):
        ss = self.__class__.__name__
        ss += f"\n{self.data}"
        return ss

    @property
    def e_reco(self):
        """Reconstructed energy axis (`~gammapy.maps.MapAxis`)"""
        return self.data.axis("energy")

    @property
    def e_true(self):
        """True energy axis (`~gammapy.maps.MapAxis`)"""
        return self.data.axis("energy_true")

    @property
    def pdf_matrix(self):
        """Energy dispersion PDF matrix (`~numpy.ndarray`).

        Rows (first index): True Energy
        Columns (second index): Reco Energy
        """
        return self.data.data.value

    def pdf_in_safe_range(self, lo_threshold, hi_threshold):
        """PDF matrix with bins outside threshold set to 0.

        Parameters
        ----------
        lo_threshold : `~astropy.units.Quantity`
            Low reco energy threshold
        hi_threshold : `~astropy.units.Quantity`
            High reco energy threshold
        """
        data = self.pdf_matrix.copy()
        energy = self.e_reco.edges

        if lo_threshold is None and hi_threshold is None:
            idx = slice(None)
        else:
            idx = (energy[:-1] < lo_threshold) | (energy[1:] > hi_threshold)
        data[:, idx] = 0
        return data

    def to_image(self, lo_threshold=None, hi_threshold=None):
        lo_threshold = lo_threshold or self.e_reco.edges[0]
        hi_threshold = hi_threshold or self.e_reco.edges[-1]
        data = self.pdf_in_safe_range(lo_threshold, hi_threshold)

        return self.__class__(
            e_reco=self.e_reco.squash(),
            e_true=self.e_true,
            data=np.sum(data, axis=1, keepdims=True),
        )

    @classmethod
    def from_energy_lo_hi(
        cls, e_true_lo, e_true_hi, e_reco_lo, e_reco_hi, data, **kwargs
    ):
        e_true_edges = edges_from_lo_hi(e_true_lo, e_true_hi)
        e_true_axis = MapAxis.from_edges(e_true_edges, interp="log", name="energy_true")

        e_reco_edges = edges_from_lo_hi(e_reco_lo, e_reco_hi)
        e_reco_axis = MapAxis.from_edges(e_reco_edges, interp="log", name="energy")

        return cls(e_true_axis, e_reco_axis, data, **kwargs)

    @classmethod
    def from_gauss(cls, e_true, e_reco, sigma, bias, pdf_threshold=1e-6):
        """Create Gaussian energy dispersion matrix (`EnergyDispersion`).

        Calls :func:`gammapy.irf.EnergyDispersion2D.from_gauss`

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`
            Bin edges of true energy axis
        e_reco : `~astropy.units.Quantity`
            Bin edges of reconstructed energy axis
        bias : float or `~numpy.ndarray`
            Center of Gaussian energy dispersion, bias
        sigma : float or `~numpy.ndarray`
            RMS width of Gaussian energy dispersion, resolution
        pdf_threshold : float, optional
            Zero suppression threshold
        """
        from .energy_dispersion import EnergyDispersion2D

        migra = np.linspace(1.0 / 3, 3, 200)
        # A dummy offset axis (need length 2 for interpolation to work)
        offset = Quantity([0, 1, 2], "deg")

        edisp = EnergyDispersion2D.from_gauss(
            e_true=e_true,
            migra=migra,
            sigma=sigma,
            bias=bias,
            offset=offset,
            pdf_threshold=pdf_threshold,
        )
        return edisp.to_energy_dispersion(offset=offset[0], e_reco=e_reco)

    @classmethod
    def from_diagonal_response(cls, e_true, e_reco=None):
        """Create energy dispersion from a diagonal response, i.e. perfect energy resolution

        This creates the matrix corresponding to a perfect energy response.
        It contains ones where the e_true center is inside the e_reco bin.
        It is a square diagonal matrix if e_true = e_reco.

        This is useful in cases where code always applies an edisp,
        but you don't want it to do anything.

        Parameters
        ----------
        e_true, e_reco : `~astropy.units.Quantity`
            Energy bounds for true and reconstructed energy axis

        Examples
        --------
        If ``e_true`` equals ``e_reco``, you get a diagonal matrix::

            e_true = [0.5, 1, 2, 4, 6] * u.TeV
            edisp = EnergyDispersion.from_diagonal_response(e_true)
            edisp.plot_matrix()

        Example with different energy binnings::

            e_true = [0.5, 1, 2, 4, 6] * u.TeV
            e_reco = [2, 4, 6] * u.TeV
            edisp = EnergyDispersion.from_diagonal_response(e_true, e_reco)
            edisp.plot_matrix()
        """
        if e_reco is None:
            e_reco = e_true

        e_true_center = 0.5 * (e_true[1:] + e_true[:-1])
        etrue_2d, ereco_lo_2d = np.meshgrid(e_true_center, e_reco[:-1])
        etrue_2d, ereco_hi_2d = np.meshgrid(e_true_center, e_reco[1:])

        data = np.logical_and(etrue_2d >= ereco_lo_2d, etrue_2d < ereco_hi_2d)
        data = np.transpose(data).astype("float")

        return cls.from_energy_lo_hi(
            e_true_lo=e_true[:-1],
            e_true_hi=e_true[1:],
            e_reco_lo=e_reco[:-1],
            e_reco_hi=e_reco[1:],
            data=data,
        )

    @classmethod
    def from_hdulist(cls, hdulist, hdu1="MATRIX", hdu2="EBOUNDS"):
        """Create `EnergyDispersion` object from `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            HDU list with ``MATRIX`` and ``EBOUNDS`` extensions.
        hdu1 : str, optional
            HDU containing the energy dispersion matrix, default: MATRIX
        hdu2 : str, optional
            HDU containing the energy axis information, default, EBOUNDS
        """
        matrix_hdu = hdulist[hdu1]
        ebounds_hdu = hdulist[hdu2]

        data = matrix_hdu.data
        header = matrix_hdu.header

        pdf_matrix = np.zeros([len(data), header["DETCHANS"]], dtype=np.float64)

        for i, l in enumerate(data):
            if l.field("N_GRP"):
                m_start = 0
                for k in range(l.field("N_GRP")):
                    pdf_matrix[
                        i,
                        l.field("F_CHAN")[k] : l.field("F_CHAN")[k]
                        + l.field("N_CHAN")[k],
                    ] = l.field("MATRIX")[m_start : m_start + l.field("N_CHAN")[k]]
                    m_start += l.field("N_CHAN")[k]

        unit = ebounds_hdu.header.get("TUNIT2")
        e_reco_lo = Quantity(ebounds_hdu.data["E_MIN"], unit=unit)
        e_reco_hi = Quantity(ebounds_hdu.data["E_MAX"], unit=unit)

        unit = matrix_hdu.header.get("TUNIT1")
        e_true_lo = Quantity(matrix_hdu.data["ENERG_LO"], unit=unit)
        e_true_hi = Quantity(matrix_hdu.data["ENERG_HI"], unit=unit)

        return cls.from_energy_lo_hi(
            e_true_lo=e_true_lo,
            e_true_hi=e_true_hi,
            e_reco_lo=e_reco_lo,
            e_reco_hi=e_reco_hi,
            data=pdf_matrix,
        )

    @classmethod
    def read(cls, filename, hdu1="MATRIX", hdu2="EBOUNDS"):
        """Read from file.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            File to read
        hdu1 : str, optional
            HDU containing the energy dispersion matrix, default: MATRIX
        hdu2 : str, optional
            HDU containing the energy axis information, default, EBOUNDS
        """
        with fits.open(make_path(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu1=hdu1, hdu2=hdu2)

    def to_hdulist(self, use_sherpa=False, **kwargs):
        """Convert RMF to FITS HDU list format.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            Header to be written in the fits file.
        energy_unit : str
            Unit in which the energy is written in the HDU list

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            RMF in HDU list format.

        Notes
        -----
        For more info on the RMF FITS file format see:
        https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html
        """
        # Cannot use table_to_fits here due to variable length array
        # http://docs.astropy.org/en/v1.0.4/io/fits/usage/unfamiliar.html

        table = self.to_table()
        name = table.meta.pop("name")

        header = fits.Header()
        header.update(table.meta)

        if use_sherpa:
            table["ENERG_HI"] = table["ENERG_HI"].quantity.to("keV")
            table["ENERG_LO"] = table["ENERG_LO"].quantity.to("keV")

        cols = table.columns
        c0 = fits.Column(
            name=cols[0].name, format="E", array=cols[0], unit=str(cols[0].unit)
        )
        c1 = fits.Column(
            name=cols[1].name, format="E", array=cols[1], unit=str(cols[1].unit)
        )
        c2 = fits.Column(name=cols[2].name, format="I", array=cols[2])
        c3 = fits.Column(name=cols[3].name, format="PI()", array=cols[3])
        c4 = fits.Column(name=cols[4].name, format="PI()", array=cols[4])
        c5 = fits.Column(name=cols[5].name, format="PE()", array=cols[5])

        hdu = fits.BinTableHDU.from_columns(
            [c0, c1, c2, c3, c4, c5], header=header, name=name
        )

        hdu_format = "ogip-sherpa" if use_sherpa else "ogip"

        ebounds_hdu = self.e_reco.to_table_hdu(format=hdu_format)
        prim_hdu = fits.PrimaryHDU()

        return fits.HDUList([prim_hdu, hdu, ebounds_hdu])

    def to_table(self):
        """Convert to `~astropy.table.Table`.

        The output table is in the OGIP RMF format.
        https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#Tab:1
        """
        rows = self.pdf_matrix.shape[0]
        n_grp = []
        f_chan = np.ndarray(dtype=np.object, shape=rows)
        n_chan = np.ndarray(dtype=np.object, shape=rows)
        matrix = np.ndarray(dtype=np.object, shape=rows)

        # Make RMF type matrix
        for i, row in enumerate(self.data.data.value):
            pos = np.nonzero(row)[0]
            borders = np.where(np.diff(pos) != 1)[0]
            # add 1 to borders for correct behaviour of np.split
            groups = np.asarray(np.split(pos, borders + 1))
            n_grp_temp = groups.shape[0] if groups.size > 0 else 1
            n_chan_temp = np.asarray([val.size for val in groups])
            try:
                f_chan_temp = np.asarray([val[0] for val in groups])
            except IndexError:
                f_chan_temp = np.zeros(1)

            n_grp.append(n_grp_temp)
            f_chan[i] = f_chan_temp
            n_chan[i] = n_chan_temp
            matrix[i] = row[pos]

        n_grp = np.asarray(n_grp, dtype=np.int16)

        # Get total number of groups and channel subsets
        numgrp, numelt = 0, 0
        for val, val2 in zip(n_grp, n_chan):
            numgrp += np.sum(val)
            numelt += np.sum(val2)

        table = Table()

        energy = self.e_true.edges
        table["ENERG_LO"] = energy[:-1]
        table["ENERG_HI"] = energy[1:]
        table["N_GRP"] = n_grp
        table["F_CHAN"] = f_chan
        table["N_CHAN"] = n_chan
        table["MATRIX"] = matrix

        table.meta = {
            "name": "MATRIX",
            "chantype": "PHA",
            "hduclass": "OGIP",
            "hduclas1": "RESPONSE",
            "hduclas2": "RSP_MATRIX",
            "detchans": self.e_reco.nbin,
            "numgrp": numgrp,
            "numelt": numelt,
            "tlmin4": 0,
        }

        return table

    def write(self, filename, use_sherpa=False, **kwargs):
        """Write to file."""
        filename = make_path(filename)
        self.to_hdulist(use_sherpa=use_sherpa).writeto(filename, **kwargs)

    def get_resolution(self, e_true):
        """Get energy resolution for a given true energy.

        The resolution is given as a percentage of the true energy

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`
            True energy
        """
        var = self._get_variance(e_true)
        idx_true = self.e_true.coord_to_idx(e_true)
        e_true_real = self.e_true.center[idx_true]
        return np.sqrt(var) / e_true_real

    def get_bias(self, e_true):
        r"""Get reconstruction bias for a given true energy.

        Bias is defined as

        .. math:: \frac{E_{reco}-E_{true}}{E_{true}}

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`
            True energy
        """
        e_reco = self.get_mean(e_true)
        idx_true = self.e_true.coord_to_idx(e_true)
        e_true_real = self.e_true.center[idx_true]
        bias = (e_reco - e_true_real) / e_true_real
        return bias

    def get_bias_energy(self, bias, emin=None, emax=None):
        """Find energy corresponding to a given bias.

        In case the solution is not unique, provide the ``emin`` or ``emax`` arguments
        to limit the solution to the given range.  By default the peak energy of the
        bias is chosen as ``emin``.

        Parameters
        ----------
        bias : float
            Bias value.
        emin : `~astropy.units.Quantity`
            Lower bracket value in case solution is not unique.
        emax : `~astropy.units.Quantity`
            Upper bracket value in case solution is not unique.

        Returns
        -------
        bias_energy : `~astropy.units.Quantity`
            Reconstructed energy corresponding to the given bias.
        """
        from gammapy.modeling.models import TemplateSpectralModel

        e_true = self.e_true.center
        values = self.get_bias(e_true)

        if emin is None:
            # use the peak bias energy as default minimum
            emin = e_true[np.nanargmax(values)]
        if emax is None:
            emax = e_true[-1]

        bias_spectrum = TemplateSpectralModel(e_true, values)
        e_true_bias = bias_spectrum.inverse(Quantity(bias), emin=emin, emax=emax)

        # return reconstructed energy
        return e_true_bias * (1 + bias)

    def get_mean(self, e_true):
        """Get mean reconstructed energy for a given true energy."""
        # find pdf for true energies
        idx = self.e_true.coord_to_idx(e_true)
        pdf = self.data.data[idx]

        # compute sum along reconstructed energy
        # axis to determine the mean
        norm = np.sum(pdf, axis=-1)
        temp = np.sum(pdf * self.e_reco.center, axis=-1)

        with np.errstate(invalid="ignore"):
            # corm can be zero
            mean = np.nan_to_num(temp / norm)

        return mean

    def _get_variance(self, e_true):
        """Get variance of log reconstructed energy."""
        # evaluate the pdf at given true energies
        idx = self.e_true.coord_to_idx(e_true)
        pdf = self.data.data[idx]

        # compute mean
        mean = self.get_mean(e_true)

        # create array of reconstructed-energy nodes
        # for each given true energy value
        # (first axis is reconstructed energy)
        erec = self.e_reco.center
        erec = np.repeat(erec, max(np.sum(mean.shape), 1)).reshape(
            erec.shape + mean.shape
        )

        # compute deviation from mean
        # (and move reconstructed energy axis to last axis)
        temp_ = (erec - mean) ** 2
        temp = np.rollaxis(temp_, 1)

        # compute sum along reconstructed energy
        # axis to determine the variance
        norm = np.sum(pdf, axis=-1)
        var = np.sum(temp * pdf, axis=-1)

        return var / norm

    def plot_matrix(self, ax=None, show_energy=None, add_cbar=False, **kwargs):
        """Plot PDF matrix.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        show_energy : `~astropy.units.Quantity`, optional
            Show energy, e.g. threshold, as vertical line
        add_cbar : bool
            Add a colorbar to the plot.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import PowerNorm

        kwargs.setdefault("cmap", "GnBu")
        norm = PowerNorm(gamma=0.5)
        kwargs.setdefault("norm", norm)

        ax = plt.gca() if ax is None else ax

        e_true = self.e_true.edges
        e_reco = self.e_reco.edges
        x = e_true.value
        y = e_reco.value
        z = self.pdf_matrix
        caxes = ax.pcolormesh(x, y, z.T, **kwargs)

        if show_energy is not None:
            ener_val = show_energy.to_value(self.reco_energy.unit)
            ax.hlines(ener_val, 0, 200200, linestyles="dashed")

        if add_cbar:
            label = "Probability density (A.U.)"
            cbar = ax.figure.colorbar(caxes, ax=ax, label=label)

        ax.set_xlabel(fr"$E_\mathrm{{True}}$ [{e_true.unit}]")
        ax.set_ylabel(fr"$E_\mathrm{{Reco}}$ [{e_reco.unit}]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        return ax

    def plot_bias(self, ax=None, **kwargs):
        """Plot reconstruction bias.

        See `~gammapy.irf.EnergyDispersion.get_bias` method.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        x = self.e_true.center.to_value("TeV")
        y = self.get_bias(self.e_true.center)

        ax.plot(x, y, **kwargs)
        ax.set_xlabel(r"$E_\mathrm{{True}}$ [TeV]")
        ax.set_ylabel(r"($E_\mathrm{{Reco}} - E_\mathrm{{True}}) / E_\mathrm{{True}}$")
        ax.set_xscale("log")
        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plot."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        self.plot_bias(ax=axes[0])
        self.plot_matrix(ax=axes[1])
        plt.tight_layout()
