# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.table import Table
from ..utils.energy import EnergyBounds, Energy
from ..utils.scripts import make_path
from ..utils.nddata import NDDataArray, BinnedDataAxis
from ..utils.fits import energy_axis_to_ebounds

__all__ = ["EnergyDispersion", "EnergyDispersion2D"]


class EnergyDispersion(object):
    """Energy dispersion matrix.

    Data format specification: :ref:`gadf:ogip-rmf`

    Parameters
    ----------
    e_true_lo, e_true_hi : `~astropy.units.Quantity`
        True energy axis binning
    e_reco_lo, e_reco_hi : `~astropy.units.Quantity`
        Reconstruced energy axis binning
    data : array_like
        2-dim energy dispersion matrix

    Examples
    --------
    Create a Gaussian energy dispersion matrix::

        import numpy as np
        import astropy.units as u
        from gammapy.irf import EnergyDispersion
        energy = np.logspace(0, 1, 101) * u.TeV
        edisp = EnergyDispersion.from_gauss(
            e_true=energy, e_reco=energy,
            sigma=0.1, bias=0,
        )

    Have a quick look:

    >>> print(edisp)
    >>> edisp.peek()

    See Also
    --------
    EnergyDispersion2D
    """

    default_interp_kwargs = dict(bounds_error=False, fill_value=0, method="nearest")
    """Default Interpolation kwargs for `~NDDataArray`. Fill zeros and do not
    interpolate"""

    def __init__(
        self,
        e_true_lo,
        e_true_hi,
        e_reco_lo,
        e_reco_hi,
        data,
        interp_kwargs=None,
        meta=None,
    ):
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs
        axes = [
            BinnedDataAxis(
                e_true_lo, e_true_hi, interpolation_mode="log", name="e_true"
            ),
            BinnedDataAxis(
                e_reco_lo, e_reco_hi, interpolation_mode="log", name="e_reco"
            ),
        ]
        self.data = NDDataArray(axes=axes, data=data, interp_kwargs=interp_kwargs)
        self.meta = OrderedDict(meta) if meta else OrderedDict()

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n{}".format(self.data)
        return ss

    def apply(self, data):
        """Apply energy dispersion.

        Computes the matrix product of ``data``
        (which typically is model flux or counts in true energy bins)
        with the energy dispersion matrix.

        Parameters
        ----------
        data : array_like
            1-dim data array.

        Returns
        -------
        convolved_data : array
            1-dim data array after multiplication with the energy dispersion matrix
        """
        if len(data) != self.e_true.nbins:
            raise ValueError(
                "Input size {} does not match true energy axis {}".format(
                    len(data), self.e_true.nbins
                )
            )
        return np.dot(data, self.data.data)

    @property
    def e_reco(self):
        """Reconstructed energy axis (`~gammapy.utils.nddata.BinnedDataAxis`)"""
        return self.data.axis("e_reco")

    @property
    def e_true(self):
        """True energy axis (`~gammapy.utils.nddata.BinnedDataAxis`)"""
        return self.data.axis("e_true")

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
        idx = np.where(
            (self.e_reco.lo < lo_threshold) | (self.e_reco.hi > hi_threshold)
        )
        data[:, idx] = 0
        return data

    @classmethod
    def from_gauss(cls, e_true, e_reco, sigma, bias, pdf_threshold=1e-6):
        """Create Gaussian energy dispersion matrix (`EnergyDispersion`).

        Calls :func:`gammapy.irf.EnergyDispersion2D.from_gauss`

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
            Bin edges of true energy axis
        e_reco : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
            Bin edges of reconstructed energy axis
        bias : float or `~numpy.ndarray`
            Center of Gaussian energy dispersion, bias
        sigma : float or `~numpy.ndarray`
            RMS width of Gaussian energy dispersion, resolution
        pdf_threshold : float, optional
            Zero suppression threshold
        """
        migra = np.linspace(1. / 3, 3, 200)
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

        return cls(
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

        e_reco = EnergyBounds.from_ebounds(ebounds_hdu)
        e_true = EnergyBounds.from_rmf_matrix(matrix_hdu)

        return cls(
            e_true_lo=e_true.lower_bounds,
            e_true_hi=e_true.upper_bounds,
            e_reco_lo=e_reco.lower_bounds,
            e_reco_hi=e_reco.upper_bounds,
            data=pdf_matrix,
        )

    @classmethod
    def read(cls, filename, hdu1="MATRIX", hdu2="EBOUNDS"):
        """Read from file.

        Parameters
        ----------
        filename : `~gammapy.extern.pathlib.Path`, str
            File to read
        hdu1 : str, optional
            HDU containing the energy dispersion matrix, default: MATRIX
        hdu2 : str, optional
            HDU containing the energy axis information, default, EBOUNDS
        """
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            edisp = cls.from_hdulist(hdulist, hdu1=hdu1, hdu2=hdu2)

        return edisp

    def to_hdulist(self, **kwargs):
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

        cols = table.columns
        c0 = fits.Column(
            name=cols[0].name, format="E", array=cols[0], unit="{}".format(cols[0].unit)
        )
        c1 = fits.Column(
            name=cols[1].name, format="E", array=cols[1], unit="{}".format(cols[1].unit)
        )
        c2 = fits.Column(name=cols[2].name, format="I", array=cols[2])
        c3 = fits.Column(name=cols[3].name, format="PI()", array=cols[3])
        c4 = fits.Column(name=cols[4].name, format="PI()", array=cols[4])
        c5 = fits.Column(name=cols[5].name, format="PE()", array=cols[5])

        hdu = fits.BinTableHDU.from_columns(
            [c0, c1, c2, c3, c4, c5], header=header, name=name
        )

        ebounds = energy_axis_to_ebounds(self.e_reco.bins)
        prim_hdu = fits.PrimaryHDU()

        return fits.HDUList([prim_hdu, hdu, ebounds])

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

        table["ENERG_LO"] = self.e_true.lo
        table["ENERG_HI"] = self.e_true.hi
        table["N_GRP"] = n_grp
        table["F_CHAN"] = f_chan
        table["N_CHAN"] = n_chan
        table["MATRIX"] = matrix

        table.meta = OrderedDict(
            [
                ("name", "MATRIX"),
                ("chantype", "PHA"),
                ("hduclass", "OGIP"),
                ("hduclas1", "RESPONSE"),
                ("hduclas2", "RSP_MATRIX"),
                ("detchans", self.e_reco.nbins),
                ("numgrp", numgrp),
                ("numelt", numelt),
                ("tlmin4", 0),
            ]
        )

        return table

    def write(self, filename, **kwargs):
        """Write to file."""
        filename = make_path(filename)
        self.to_hdulist().writeto(str(filename), **kwargs)

    def get_resolution(self, e_true):
        """Get energy resolution for a given true energy.

        The resolution is given as a percentage of the true energy

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`
            True energy
        """
        var = self._get_variance(e_true)
        idx_true = self.e_true.find_node(e_true)
        e_true_real = self.e_true.nodes[idx_true]
        return np.sqrt(var) / e_true_real

    def get_bias(self, e_true):
        r"""Get reconstruction bias for a given true energy.

        Bias is defined as

        .. math::

            \frac{E_{reco}-E_{true}}{E_{true}}

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`
            True energy
        """
        e_reco = self.get_mean(e_true)
        idx_true = self.e_true.find_node(e_true)
        e_true_real = self.e_true.nodes[idx_true]
        bias = (e_reco - e_true_real) / e_true_real
        return bias

    def get_mean(self, e_true):
        """Get mean reconstructed energy for a given true energy."""
        # find pdf for true energies
        idx = self.e_true.find_node(e_true)
        pdf = self.data.data[idx]

        # compute sum along reconstructed energy
        # axis to determine the mean
        norm = np.sum(pdf, axis=-1)
        temp = np.sum(pdf * self.e_reco.nodes, axis=-1)

        return temp / norm

    def _get_variance(self, e_true):
        """Get variance of log reconstructed energy."""
        # evaluate the pdf at given true energies
        idx = self.e_true.find_node(e_true)
        pdf = self.data.data[idx]

        # compute mean
        mean = self.get_mean(e_true)

        # create array of reconstructed-energy nodes
        # for each given true energy value
        # (first axis is reconstructed energy)
        erec = self.e_reco.nodes
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

    def to_sherpa(self, name):
        """Convert to `sherpa.astro.data.DataRMF`.

        Parameters
        ----------
        name : str
            Instance name
        """
        from sherpa.astro.data import DataRMF
        from sherpa.utils import SherpaUInt, SherpaFloat

        # Need to modify RMF data
        # see https://github.com/sherpa/sherpa/blob/master/sherpa/astro/io/pyfits_backend.py#L727

        table = self.to_table()
        n_grp = table["N_GRP"].data.astype(SherpaUInt)
        f_chan = table["F_CHAN"].data
        n_chan = table["N_CHAN"].data
        matrix = table["MATRIX"].data

        good = n_grp > 0
        matrix = matrix[good]
        matrix = np.concatenate([row for row in matrix])
        matrix = matrix.astype(SherpaFloat)

        good = n_grp > 0
        f_chan = f_chan[good]
        f_chan = np.concatenate([row for row in f_chan]).astype(SherpaUInt)
        n_chan = n_chan[good]
        n_chan = np.concatenate([row for row in n_chan]).astype(SherpaUInt)

        return DataRMF(
            name=name,
            energ_lo=table["ENERG_LO"].quantity.to("keV").value.astype(SherpaFloat),
            energ_hi=table["ENERG_HI"].quantity.to("keV").value.astype(SherpaFloat),
            matrix=matrix,
            n_grp=n_grp,
            n_chan=n_chan,
            f_chan=f_chan,
            detchans=self.e_reco.nbins,
            e_min=self.e_reco.lo.to("keV").value,
            e_max=self.e_reco.hi.to("keV").value,
            offset=0,
        )

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

        e_true = self.e_true.bins
        e_reco = self.e_reco.bins
        x = e_true.value
        y = e_reco.value
        z = self.pdf_matrix
        caxes = ax.pcolormesh(x, y, z.T, **kwargs)

        if show_energy is not None:
            ener_val = Quantity(show_energy).to(self.reco_energy.unit).value
            ax.hlines(ener_val, 0, 200200, linestyles="dashed")

        if add_cbar:
            label = "Probability density (A.U.)"
            cbar = ax.figure.colorbar(caxes, ax=ax, label=label)

        ax.set_xlabel("$E_\mathrm{{True}}$ [{unit}]".format(unit=e_true.unit))
        ax.set_ylabel("$E_\mathrm{{Reco}}$ [{unit}]".format(unit=e_reco.unit))
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

        x = self.e_true.nodes.to("TeV").value
        y = self.get_bias(self.e_true.nodes)

        ax.plot(x, y, **kwargs)
        ax.set_xlabel("$E_\mathrm{{True}}$ [TeV]")
        ax.set_ylabel(r"($E_\mathrm{{True}} - E_\mathrm{{Reco}} / E_\mathrm{{True}}$)")
        ax.set_xscale("log")
        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plot."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        self.plot_bias(ax=axes[0])
        self.plot_matrix(ax=axes[1])
        plt.tight_layout()


class EnergyDispersion2D(object):
    """Offset-dependent energy dispersion matrix.

    Data format specification: :ref:`gadf:edisp_2d`

    Parameters
    ----------
    e_true_lo, e_true_hi : `~astropy.units.Quantity`
        True energy axis binning
    migra_lo, migra_hi : `~numpy.ndarray`
        Energy migration axis binning
    offset_lo, offset_hi : `~astropy.coordinates.Angle`
        Field of view offset axis binning
    data : `~numpy.ndarray`
        Energy dispersion probability density

    Examples
    --------
    Read energy dispersion IRF from disk:

    >>> from gammapy.irf import EnergyDispersion2D
    >>> from gammapy.utils.energy import EnergyBounds
    >>> filename = '$GAMMAPY_EXTRA/test_datasets/irf/hess/pa/hess_edisp_2d_023523.fits.gz'
    >>> edisp2d = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
    >>> print(edisp2d)
    EnergyDispersion2D
    NDDataArray summary info
    e_true         : size =    15, min =  0.125 TeV, max = 80.000 TeV
    migra          : size =   100, min =  0.051, max = 10.051
    offset         : size =     6, min =  0.125 deg, max =  2.500 deg
    Data           : size =  9000, min =  0.000, max =  3.405

    Create energy dispersion matrix (`~gammapy.irf.EnergyDispersion`)
    for a given field of view offset and energy binning:

    >>> energy = EnergyBounds.equal_log_spacing(0.1,20,60, 'TeV')
    >>> offset = '1.2 deg'
    >>> edisp = edisp2d.to_energy_dispersion(offset=offset, e_reco=energy, e_true=energy)
    >>> print(edisp)
    EnergyDispersion
    NDDataArray summary info
    e_true         : size =    60, min =  0.105 TeV, max = 19.136 TeV
    e_reco         : size =    60, min =  0.105 TeV, max = 19.136 TeV
    Data           : size =  3600, min =  0.000, max =  0.266

    See Also
    --------
    EnergyDispersion
    """

    default_interp_kwargs = dict(bounds_error=False, fill_value=None)
    """Default Interpolation kwargs for `~gammapy.utils.nddata.NDDataArray`. Extrapolate."""

    def __init__(
        self,
        e_true_lo,
        e_true_hi,
        migra_lo,
        migra_hi,
        offset_lo,
        offset_hi,
        data,
        interp_kwargs=None,
        meta=None,
    ):
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs
        axes = [
            BinnedDataAxis(
                e_true_lo, e_true_hi, interpolation_mode="log", name="e_true"
            ),
            BinnedDataAxis(
                migra_lo, migra_hi, interpolation_mode="linear", name="migra"
            ),
            BinnedDataAxis(
                offset_lo, offset_hi, interpolation_mode="linear", name="offset"
            ),
        ]
        self.data = NDDataArray(axes=axes, data=data, interp_kwargs=interp_kwargs)
        self.meta = OrderedDict(meta) if meta else OrderedDict()

    def __str__(self):
        ss = self.__class__.__name__
        ss += "\n{}".format(self.data)
        return ss

    @classmethod
    def from_gauss(cls, e_true, migra, bias, sigma, offset, pdf_threshold=1e-6):
        """Create Gaussian energy dispersion matrix (`EnergyDispersion2D`).

        The output matrix will be Gaussian in (e_true / e_reco).

        The ``bias`` and ``sigma`` should be either floats or arrays of same dimension than
        ``e_true``. ``bias`` refers to the mean value of the ``migra``
        distribution minus one, i.e. ``bias=0`` means no bias.

        Note that, the output matrix is flat in offset.

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
            Bin edges of true energy axis
        migra : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
            Bin edges of migra axis
        bias : float or `~numpy.ndarray`
            Center of Gaussian energy dispersion, bias
        sigma : float or `~numpy.ndarray`
            RMS width of Gaussian energy dispersion, resolution
        offset : `~astropy.units.Quantity`, `~gammapy.utils.nddata.BinnedDataAxis`
            Bin edges of offset
        pdf_threshold : float, optional
            Zero suppression threshold
        """
        from scipy.special import erf

        e_true = EnergyBounds(e_true)
        # erf does not work with Quantities
        true = e_true.log_centers.to("TeV").value

        true2d, migra2d = np.meshgrid(true, migra)

        migra2d_lo = migra2d[:-1, :]
        migra2d_hi = migra2d[1:, :]

        # Analytical formula for integral of Gaussian
        s = np.sqrt(2) * sigma
        t1 = (migra2d_hi - 1 - bias) / s
        t2 = (migra2d_lo - 1 - bias) / s
        pdf = (erf(t1) - erf(t2)) / 2

        pdf_array = pdf.T[:, :, np.newaxis] * np.ones(len(offset) - 1)

        pdf_array = np.where(pdf_array > pdf_threshold, pdf_array, 0)

        return cls(
            e_true[:-1],
            e_true[1:],
            migra[:-1],
            migra[1:],
            offset[:-1],
            offset[1:],
            pdf_array,
        )

    @classmethod
    def from_table(cls, table):
        """Create from `~astropy.table.Table`."""
        if "ENERG_LO" in table.colnames:
            e_lo = table["ENERG_LO"].quantity[0]
            e_hi = table["ENERG_HI"].quantity[0]
        elif "ETRUE_LO" in table.colnames:
            e_lo = table["ETRUE_LO"].quantity[0]
            e_hi = table["ETRUE_HI"].quantity[0]
        else:
            raise ValueError(
                'Invalid column names. Need "ENERG_LO/ENERG_HI" or "ETRUE_LO/ETRUE_HI"'
            )
        o_lo = table["THETA_LO"].quantity[0]
        o_hi = table["THETA_HI"].quantity[0]
        m_lo = table["MIGRA_LO"].quantity[0]
        m_hi = table["MIGRA_HI"].quantity[0]

        matrix = (
            table["MATRIX"].quantity[0].transpose()
        )  ## TODO Why does this need to be transposed?
        return cls(
            e_true_lo=e_lo,
            e_true_hi=e_hi,
            offset_lo=o_lo,
            offset_hi=o_hi,
            migra_lo=m_lo,
            migra_hi=m_hi,
            data=matrix,
        )

    @classmethod
    def from_hdulist(cls, hdulist, hdu="edisp_2d"):
        """Create from `~astropy.io.fits.HDUList`."""
        return cls.from_table(Table.read(hdulist[hdu]))

    @classmethod
    def read(cls, filename, hdu="edisp_2d"):
        """Read from FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        filename = make_path(filename)
        with fits.open(str(filename), memmap=False) as hdulist:
            edisp = cls.from_hdulist(hdulist, hdu)

        return edisp

    def to_energy_dispersion(self, offset, e_true=None, e_reco=None):
        """Detector response R(Delta E_reco, Delta E_true)

        Probability to reconstruct an energy in a given true energy band
        in a given reconstructed energy band

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset
        e_true : `~gammapy.utils.energy.EnergyBounds`, None
            True energy axis
        e_reco : `~gammapy.utils.energy.EnergyBounds`
            Reconstructed energy axis

        Returns
        -------
        edisp : `~gammapy.irf.EnergyDispersion`
            Energy dispersion matrix
        """
        offset = Angle(offset)
        e_true = self.data.axis("e_true").bins if e_true is None else e_true
        e_reco = self.data.axis("e_true").bins if e_reco is None else e_reco
        e_true = EnergyBounds(e_true)
        e_reco = EnergyBounds(e_reco)

        data = []
        for energy in e_true.log_centers:
            vec = self.get_response(offset=offset, e_true=energy, e_reco=e_reco)
            data.append(vec)

        data = np.asarray(data)
        e_lo, e_hi = e_true[:-1], e_true[1:]
        ereco_lo, ereco_hi = (e_reco[:-1], e_reco[1:])

        return EnergyDispersion(
            e_true_lo=e_lo,
            e_true_hi=e_hi,
            e_reco_lo=ereco_lo,
            e_reco_hi=ereco_hi,
            data=data,
        )

    def get_response(self, offset, e_true, e_reco=None, migra_step=5e-3):
        """Detector response R(Delta E_reco, E_true)

        Probability to reconstruct a given true energy in a given reconstructed
        energy band. In each reco bin, you integrate with a riemann sum over
        the default migra bin of your analysis.

        Parameters
        ----------
        e_true : `~gammapy.utils.energy.Energy`
            True energy
        e_reco : `~gammapy.utils.energy.EnergyBounds`, None
            Reconstructed energy axis
        offset : `~astropy.coordinates.Angle`
            Offset
        migra_step : float
            Integration step in migration

        Returns
        -------
        rv : `~numpy.ndarray`
            Redistribution vector
        """
        e_true = Energy(e_true)

        if e_reco is None:
            # Default: e_reco nodes = migra nodes * e_true nodes
            migra_axis = self.data.axis("migra")
            e_reco = EnergyBounds.from_lower_and_upper_bounds(
                migra_axis.lo * e_true, migra_axis.hi * e_true
            )
        else:
            # Translate given e_reco binning to migra at bin center
            e_reco = EnergyBounds(e_reco)

        # migration value of e_reco bounds
        migra_e_reco = e_reco / e_true

        # Define a vector of migration with mig_step step
        mrec_min = self.data.axis("migra").lo[0]
        mrec_max = self.data.axis("migra").hi[-1]
        mig_array = np.arange(mrec_min, mrec_max, migra_step)

        # Compute energy dispersion probability dP/dm for each element of migration array
        vals = self.data.evaluate(offset=offset, e_true=e_true, migra=mig_array)

        # Compute normalized cumulative sum to prepare integration
        with np.errstate(invalid="ignore"):
            tmp = np.nan_to_num(np.cumsum(vals) / np.sum(vals))

        # Determine positions (bin indices) of e_reco bounds in migration array
        pos_mig = np.digitize(migra_e_reco, mig_array) - 1
        # We ensure that no negative values are found
        pos_mig = np.maximum(pos_mig, 0)

        # We compute the difference between 2 successive bounds in e_reco
        # to get integral over reco energy bin
        integral = np.diff(tmp[pos_mig])

        return integral

    def plot_migration(self, ax=None, offset=None, e_true=None, migra=None, **kwargs):
        """Plot energy dispersion for given offset and true energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`, optional
            Offset
        e_true : `~gammapy.utils.energy.Energy`, optional
            True energy
        migra : `~numpy.array`, list, optional
            Migration nodes

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset = Angle([1], "deg")
        else:
            offset = np.atleast_1d(Angle(offset))
        if e_true is None:
            e_true = Energy([0.1, 1, 10], "TeV")
        else:
            e_true = np.atleast_1d(Energy(e_true))
        migra = self.data.axis("migra").nodes if migra is None else migra

        for ener in e_true:
            for off in offset:
                disp = self.data.evaluate(offset=off, e_true=ener, migra=migra)
                label = "offset = {0:.1f}\nenergy = {1:.1f}".format(off, ener)
                ax.plot(migra, disp, label=label, **kwargs)

        ax.set_xlabel("$E_\mathrm{{Reco}} / E_\mathrm{{True}}$")
        ax.set_ylabel("Probability density")
        ax.legend(loc="upper left")

        return ax

    def plot_bias(self, ax=None, offset=None, add_cbar=False, **kwargs):
        """Plot migration as a function of true energy for a given offset.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`, optional
            Offset
        add_cbar : bool
            Add a colorbar to the plot.
        kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axis
        """
        from matplotlib.colors import PowerNorm
        import matplotlib.pyplot as plt

        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("norm", PowerNorm(gamma=0.5))

        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset = Angle([1], "deg")

        e_true = self.data.axis("e_true").bins
        migra = self.data.axis("migra").bins

        x = e_true.value
        y = migra.value
        z = self.data.evaluate(offset=offset, e_true=e_true, migra=migra).value

        caxes = ax.pcolormesh(x, y, z.T, **kwargs)

        if add_cbar:
            label = "Probability density (A.U.)"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        ax.set_xlabel("$E_\mathrm{{True}}$ [{unit}]".format(unit=e_true.unit))
        ax.set_ylabel("$E_\mathrm{{Reco}} / E_\mathrm{{True}}$")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xscale("log")
        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : (float, float)
            Size of the resulting plot
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.plot_bias(ax=axes[0])
        self.plot_migration(ax=axes[1])
        edisp = self.to_energy_dispersion(offset="1 deg")
        edisp.plot_matrix(ax=axes[2])

        plt.tight_layout()

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        meta = self.meta.copy()
        table = Table(meta=meta)
        table["ENERG_LO"] = self.data.axis("e_true").lo[np.newaxis]
        table["ENERG_HI"] = self.data.axis("e_true").hi[np.newaxis]
        table["MIGRA_LO"] = self.data.axis("migra").hi[np.newaxis]
        table["MIGRA_HI"] = self.data.axis("migra").hi[np.newaxis]
        table["THETA_LO"] = self.data.axis("offset").lo[np.newaxis]
        table["THETA_HI"] = self.data.axis("offset").hi[np.newaxis]
        table["MATRIX"] = self.data.data.T[np.newaxis]
        return table

    def to_fits(self, name="ENERGY DISPERSION"):
        """Convert to `~astropy.io.fits.BinTable`."""
        return fits.BinTableHDU(self.to_table(), name=name)
