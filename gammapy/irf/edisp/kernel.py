# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from gammapy.maps import MapAxis
from gammapy.utils.scripts import make_path
from ..core import IRF

__all__ = ["EDispKernel"]


class EDispKernel(IRF):
    """Energy dispersion matrix.

    Data format specification: :ref:`gadf:ogip-rmf`

    Parameters
    ----------
    energy_axis_true : `~gammapy.maps.MapAxis`
        True energy axis. Its name must be "energy_true"
    energy_axis : `~gammapy.maps.MapAxis`
        Reconstructed energy axis. Its name must be "energy"
    data : array_like
        2-dim energy dispersion matrix

    Examples
    --------
    Create a Gaussian energy dispersion matrix::

    >>> from gammapy.maps import MapAxis
    >>> from gammapy.irf import EDispKernel
    >>> energy = MapAxis.from_energy_bounds(0.1,10,10, unit='TeV')
    >>> energy_true = MapAxis.from_energy_bounds(0.1,10,10, unit='TeV', name='energy_true')
    >>> edisp = EDispKernel.from_gauss(
    >>>     energy_axis_true=energy_true, energy_axis=energy, sigma=0.1, bias=0
    >>> )

    Have a quick look:

    >>> print(edisp)
    EDispKernel
    -----------
    <BLANKLINE>
      axes  : ['energy_true', 'energy']
      shape : (10, 10)
      ndim  : 2
      unit  :
      dtype : float64
    <BLANKLINE>
    >>> edisp.peek()

    """

    tag = "edisp_kernel"
    required_axes = ["energy_true", "energy"]
    default_interp_kwargs = dict(bounds_error=False, fill_value=0, method="nearest")
    """Default Interpolation kwargs for `~IRF`. Fill zeros and do not
    interpolate"""

    @property
    def pdf_matrix(self):
        """Energy dispersion PDF matrix (`~numpy.ndarray`).

        Rows (first index): True Energy
        Columns (second index): Reco Energy
        """
        return self.data

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
        energy = self.axes["energy"].edges

        if lo_threshold is None and hi_threshold is None:
            idx = slice(None)
        else:
            idx = (energy[:-1] < lo_threshold) | (energy[1:] > hi_threshold)
        data[:, idx] = 0
        return data

    def to_image(self, lo_threshold=None, hi_threshold=None):
        """Return a 2D edisp by summing the pdf matrix over the ereco axis.

        Parameters
        ----------
        lo_threshold : `~astropy.units.Quantity`, optional
            Low reco energy threshold
        hi_threshold : `~astropy.units.Quantity`, optional
            High reco energy threshold
        """
        energy_axis = self.axes["energy"]
        lo_threshold = lo_threshold or energy_axis.edges[0]
        hi_threshold = hi_threshold or energy_axis.edges[-1]
        data = self.pdf_in_safe_range(lo_threshold, hi_threshold)

        return self.__class__(
            axes=self.axes.squash("energy"),
            data=np.sum(data, axis=1, keepdims=True),
        )

    @classmethod
    def from_gauss(cls, energy_axis_true, energy_axis, sigma, bias, pdf_threshold=1e-6):
        """Create Gaussian energy dispersion matrix (`EnergyDispersion`).

        Calls :func:`gammapy.irf.EnergyDispersion2D.from_gauss`

        Parameters
        ----------
        energy_axis_true : `~astropy.units.Quantity`
            Bin edges of true energy axis
        energy_axis : `~astropy.units.Quantity`
            Bin edges of reconstructed energy axis
        bias : float or `~numpy.ndarray`
            Center of Gaussian energy dispersion, bias
        sigma : float or `~numpy.ndarray`
            RMS width of Gaussian energy dispersion, resolution
        pdf_threshold : float, optional
            Zero suppression threshold

        Returns
        -------
        edisp : `EDispKernel`
            Edisp kernel.
        """
        from .core import EnergyDispersion2D

        migra_axis = MapAxis.from_bounds(1.0 / 3, 3, nbin=200, name="migra")

        # A dummy offset axis (need length 2 for interpolation to work)
        offset_axis = MapAxis.from_edges([0, 1, 2], unit="deg", name="offset")

        edisp = EnergyDispersion2D.from_gauss(
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            offset_axis=offset_axis,
            sigma=sigma,
            bias=bias,
            pdf_threshold=pdf_threshold,
        )
        return edisp.to_edisp_kernel(
            offset=offset_axis.center[0], energy=energy_axis.edges
        )

    @classmethod
    def from_diagonal_response(cls, energy_axis_true, energy_axis=None):
        """Create energy dispersion from a diagonal response, i.e. perfect energy resolution

        This creates the matrix corresponding to a perfect energy response.
        It contains ones where the energy_true center is inside the e_reco bin.
        It is a square diagonal matrix if energy_true = e_reco.

        This is useful in cases where code always applies an edisp,
        but you don't want it to do anything.

        Parameters
        ----------
        energy_axis_true, energy_axis : `MapAxis`
            True and reconstructed energy axis

        Examples
        --------
        If ``energy_true`` equals ``energy``, you get a diagonal matrix::

            from gammapy.irf import EDispKernel
            from gammapy.maps import MapAxis

            energy_true_axis = MapAxis.from_energy_edges(
                    [0.5, 1, 2, 4, 6] * u.TeV, name="energy_true"
                )
            edisp = EDispKernel.from_diagonal_response(energy_true_axis)
            edisp.plot_matrix()

        Example with different energy binnings::

            energy_true_axis = MapAxis.from_energy_edges(
                    [0.5, 1, 2, 4, 6] * u.TeV, name="energy_true"
                )
            energy_axis = MapAxis.from_energy_edges([2, 4, 6] * u.TeV)
            edisp = EDispKernel.from_diagonal_response(energy_true_axis, energy_axis)
            edisp.plot_matrix()
        """
        from .map import get_overlap_fraction

        energy_axis_true.assert_name("energy_true")

        if energy_axis is None:
            energy_axis = energy_axis_true.copy(name="energy")

        data = get_overlap_fraction(energy_axis, energy_axis_true)
        return cls(axes=[energy_axis_true, energy_axis], data=data.value)

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
                    chan_min = l.field("F_CHAN")[k]
                    chan_max = l.field("F_CHAN")[k] + l.field("N_CHAN")[k]

                    pdf_matrix[i, chan_min:chan_max] = l.field("MATRIX")[
                        m_start : m_start + l.field("N_CHAN")[k]  # noqa: E203
                    ]
                    m_start += l.field("N_CHAN")[k]

        table = Table.read(ebounds_hdu)
        energy_axis = MapAxis.from_table(table, format="ogip")

        table = Table.read(matrix_hdu)
        energy_axis_true = MapAxis.from_table(table, format="ogip-arf")

        return cls(axes=[energy_axis_true, energy_axis], data=pdf_matrix)

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
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu1=hdu1, hdu2=hdu2)

    def to_hdulist(self, format="ogip", **kwargs):
        """Convert RMF to FITS HDU list format.

        Parameters
        ----------
        format : {"ogip", "ogip-sherpa"}
            Format to use.

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
        format_arf = format.replace("ogip", "ogip-arf")
        table = self.to_table(format=format_arf)

        name = table.meta.pop("name")

        header = fits.Header()
        header.update(table.meta)

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

        ebounds_hdu = self.axes["energy"].to_table_hdu(format=format)
        prim_hdu = fits.PrimaryHDU()

        return fits.HDUList([prim_hdu, hdu, ebounds_hdu])

    def to_table(self, format="ogip"):
        """Convert to `~astropy.table.Table`.

        The output table is in the OGIP RMF format.
        https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#Tab:1  # noqa: E501

        Parameters
        ----------
        format : {"ogip", "ogip-sherpa"}
            Format to use.

        Returns
        -------
        table : `~astropy.table.Table`
            Matrix table

        """
        table = self.axes["energy_true"].to_table(format=format)

        rows = self.pdf_matrix.shape[0]
        n_grp = []
        f_chan = np.ndarray(dtype=np.object, shape=rows)
        n_chan = np.ndarray(dtype=np.object, shape=rows)
        matrix = np.ndarray(dtype=np.object, shape=rows)

        # Make RMF type matrix
        for idx, row in enumerate(self.data):
            pos = np.nonzero(row)[0]
            borders = np.where(np.diff(pos) != 1)[0]
            # add 1 to borders for correct behaviour of np.split
            groups = np.split(pos, borders + 1)
            n_grp_temp = len(groups) if len(groups) > 0 else 1
            n_chan_temp = np.asarray([val.size for val in groups])
            try:
                f_chan_temp = np.asarray([val[0] for val in groups])
            except IndexError:
                f_chan_temp = np.zeros(1)

            n_grp.append(n_grp_temp)
            f_chan[idx] = f_chan_temp
            n_chan[idx] = n_chan_temp
            matrix[idx] = row[pos]

        n_grp = np.asarray(n_grp, dtype=np.int16)

        # Get total number of groups and channel subsets
        numgrp, numelt = 0, 0
        for val, val2 in zip(n_grp, n_chan):
            numgrp += np.sum(val)
            numelt += np.sum(val2)

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
            "detchans": self.axes["energy"].nbin,
            "numgrp": numgrp,
            "numelt": numelt,
            "tlmin4": 0,
        }

        return table

    def write(self, filename, format="ogip", **kwargs):
        """Write to file.

        Parameters
        ----------
        filename : str
            Filename
        format : {"ogip", "ogip-sherpa"}
            Format to use.

        """
        filename = str(make_path(filename))
        self.to_hdulist(format=format).writeto(filename, **kwargs)

    def get_resolution(self, energy_true):
        """Get energy resolution for a given true energy.

        The resolution is given as a percentage of the true energy

        Parameters
        ----------
        energy_true : `~astropy.units.Quantity`
            True energy
        """
        energy_axis_true = self.axes["energy_true"]
        var = self._get_variance(energy_true)
        idx_true = energy_axis_true.coord_to_idx(energy_true)
        energy_true_real = energy_axis_true.center[idx_true]
        return np.sqrt(var) / energy_true_real

    def get_bias(self, energy_true):
        r"""Get reconstruction bias for a given true energy.

        Bias is defined as

        .. math:: \frac{E_{reco}-E_{true}}{E_{true}}

        Parameters
        ----------
        energy_true : `~astropy.units.Quantity`
            True energy
        """
        energy_axis_true = self.axes["energy_true"]
        energy = self.get_mean(energy_true)
        idx_true = energy_axis_true.coord_to_idx(energy_true)
        energy_true_real = energy_axis_true.center[idx_true]
        bias = (energy - energy_true_real) / energy_true_real
        return bias

    def get_bias_energy(self, bias, energy_min=None, energy_max=None):
        """Find energy corresponding to a given bias.

        In case the solution is not unique, provide the ``energy_min`` or ``energy_max`` arguments
        to limit the solution to the given range.  By default the peak energy of the
        bias is chosen as ``energy_min``.

        Parameters
        ----------
        bias : float
            Bias value.
        energy_min : `~astropy.units.Quantity`
            Lower bracket value in case solution is not unique.
        energy_max : `~astropy.units.Quantity`
            Upper bracket value in case solution is not unique.

        Returns
        -------
        bias_energy : `~astropy.units.Quantity`
            Reconstructed energy corresponding to the given bias.
        """
        from gammapy.modeling.models import TemplateSpectralModel

        energy_true = self.axes["energy_true"].center
        values = self.get_bias(energy_true)

        if energy_min is None:
            # use the peak bias energy as default minimum
            energy_min = energy_true[np.nanargmax(values)]
        if energy_max is None:
            energy_max = energy_true[-1]

        bias_spectrum = TemplateSpectralModel(energy=energy_true, values=values)

        energy_true_bias = bias_spectrum.inverse(
            Quantity(bias), energy_min=energy_min, energy_max=energy_max
        )
        if np.isnan(energy_true_bias[0]):
            energy_true_bias[0] = energy_min
        # return reconstructed energy
        return energy_true_bias * (1 + bias)

    def get_mean(self, energy_true):
        """Get mean reconstructed energy for a given true energy."""
        idx = self.axes["energy_true"].coord_to_idx(energy_true)
        pdf = self.data[idx]

        # compute sum along reconstructed energy
        norm = np.sum(pdf, axis=-1)
        temp = np.sum(pdf * self.axes["energy"].center, axis=-1)

        with np.errstate(invalid="ignore"):
            # corm can be zero
            mean = np.nan_to_num(temp / norm)

        return mean

    def _get_variance(self, energy_true):
        """Get variance of log reconstructed energy."""
        # evaluate the pdf at given true energies
        idx = self.axes["energy_true"].coord_to_idx(energy_true)
        pdf = self.data[idx]

        # compute mean
        mean = self.get_mean(energy_true)

        # create array of reconstructed-energy nodes
        # for each given true energy value
        # (first axis is reconstructed energy)
        erec = self.axes["energy"].center
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

    def plot_matrix(self, ax=None, add_cbar=False, **kwargs):
        """Plot PDF matrix.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        add_cbar : bool
            Add a colorbar to the plot.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        kwargs.setdefault("cmap", "GnBu")
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)
        kwargs.setdefault("norm", norm)

        ax = plt.gca() if ax is None else ax

        energy_axis_true = self.axes["energy_true"]
        energy_axis = self.axes["energy"]

        with quantity_support():
            caxes = ax.pcolormesh(
                energy_axis_true.edges, energy_axis.edges, self.data.T, **kwargs
            )

        if add_cbar:
            label = "Probability density (A.U.)"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        energy_axis_true.format_plot_xaxis(ax=ax)
        energy_axis.format_plot_yaxis(ax=ax)
        return ax

    def plot_bias(self, ax=None, **kwargs):
        """Plot reconstruction bias.

        See `~gammapy.irf.EnergyDispersion.get_bias` method.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Plot axis
        **kwargs : dict
            Kwyrow

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Plot axis
        """
        ax = plt.gca() if ax is None else ax

        energy = self.axes["energy_true"].center
        bias = self.get_bias(energy)

        with quantity_support():
            ax.plot(energy, bias, **kwargs)

        ax.set_xlabel(f"$E_\\mathrm{{True}}$ [{ax.yaxis.units}]")
        ax.set_ylabel(
            "($E_\\mathrm{{Reco}} - E_\\mathrm{{True}}) / E_\\mathrm{{True}}$"
        )
        ax.set_xscale("log")
        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : tuple
            Size of the figure.

        """
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        self.plot_bias(ax=axes[0])
        self.plot_matrix(ax=axes[1])
        plt.tight_layout()
