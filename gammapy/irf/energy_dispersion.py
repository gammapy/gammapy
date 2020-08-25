# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.special
from scipy.interpolate import interp1d
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from gammapy.maps import MapAxis
from gammapy.maps.utils import edges_from_lo_hi
from gammapy.utils.nddata import NDDataArray
from gammapy.utils.scripts import make_path
from .edisp_kernel import EDispKernel

__all__ = ["EnergyDispersion2D"]


class EnergyDispersion2D:
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

    >>> from gammapy.maps import MapAxis
    >>> from gammapy.irf import EnergyDispersion2D
    >>> filename = '$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz'
    >>> edisp2d = EnergyDispersion2D.read(filename, hdu="EDISP")

    Create energy dispersion matrix (`~gammapy.irf.EnergyDispersion`)
    for a given field of view offset and energy binning:

    >>> energy = MapAxis.from_bounds(0.1, 20, nbin=60, unit="TeV", interp="log").edges
    >>> edisp = edisp2d.to_energy_dispersion(offset='1.2 deg', e_reco=energy, e_true=energy)

    See Also
    --------
    EnergyDispersion
    """
    tag = "edisp_2d"
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

        e_true_edges = edges_from_lo_hi(e_true_lo, e_true_hi)
        e_true_axis = MapAxis.from_edges(e_true_edges, interp="log", name="energy_true")

        migra_edges = edges_from_lo_hi(migra_lo, migra_hi)
        migra_axis = MapAxis.from_edges(
            migra_edges, interp="lin", name="migra", unit=""
        )

        # TODO: for some reason the H.E.S.S. DL3 files contain the same values for offset_hi and offset_lo
        if np.allclose(offset_lo.to_value("deg"), offset_hi.to_value("deg")):
            offset_axis = MapAxis.from_nodes(offset_lo, interp="lin", name="offset")
        else:
            offset_edges = edges_from_lo_hi(offset_lo, offset_hi)
            offset_axis = MapAxis.from_edges(offset_edges, interp="lin", name="offset")

        axes = [e_true_axis, migra_axis, offset_axis]

        self.data = NDDataArray(axes=axes, data=data, interp_kwargs=interp_kwargs)
        self.meta = meta or {}

    def __str__(self):
        ss = self.__class__.__name__
        ss += f"\n{self.data}"
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
        e_true : `~astropy.units.Quantity`
            Bin edges of true energy axis
        migra : `~astropy.units.Quantity`
            Bin edges of migra axis
        bias : float or `~numpy.ndarray`
            Center of Gaussian energy dispersion, bias
        sigma : float or `~numpy.ndarray`
            RMS width of Gaussian energy dispersion, resolution
        offset : `~astropy.units.Quantity`
            Bin edges of offset
        pdf_threshold : float, optional
            Zero suppression threshold
        """
        e_true = Quantity(e_true)
        # erf does not work with Quantities
        true = MapAxis.from_edges(e_true, interp="log").center.to_value("TeV")

        true2d, migra2d = np.meshgrid(true, migra)

        migra2d_lo = migra2d[:-1, :]
        migra2d_hi = migra2d[1:, :]

        # Analytical formula for integral of Gaussian
        s = np.sqrt(2) * sigma
        t1 = (migra2d_hi - 1 - bias) / s
        t2 = (migra2d_lo - 1 - bias) / s
        pdf = (scipy.special.erf(t1) - scipy.special.erf(t2)) / 2

        pdf_array = pdf.T[:, :, np.newaxis] * np.ones(len(offset) - 1)

        pdf_array[pdf_array < pdf_threshold] = 0

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

        # TODO Why does this need to be transposed?
        matrix = table["MATRIX"].quantity[0].transpose()

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
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, hdu)

    def to_energy_dispersion(self, offset, e_true=None, e_reco=None):
        """Detector response R(Delta E_reco, Delta E_true)

        Probability to reconstruct an energy in a given true energy band
        in a given reconstructed energy band

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset
        e_true : `~astropy.units.Quantity`, None
            True energy axis
        e_reco : `~astropy.units.Quantity`
            Reconstructed energy axis

        Returns
        -------
        edisp : `~gammapy.irf.EnergyDispersion`
            Energy dispersion matrix
        """
        offset = Angle(offset)
        e_true = self.data.axis("energy_true").edges if e_true is None else e_true
        e_reco = self.data.axis("energy_true").edges if e_reco is None else e_reco

        data = []
        for energy in MapAxis.from_edges(e_true, interp="log").center:
            vec = self.get_response(offset=offset, e_true=energy, e_reco=e_reco)
            data.append(vec)

        data = np.asarray(data)
        e_lo, e_hi = e_true[:-1], e_true[1:]
        ereco_lo, ereco_hi = (e_reco[:-1], e_reco[1:])

        return EDispKernel.from_energy_lo_hi(
            e_true_lo=e_lo,
            e_true_hi=e_hi,
            e_reco_lo=ereco_lo,
            e_reco_hi=ereco_hi,
            data=data,
        )

    def get_response(self, offset, e_true, e_reco=None):
        """Detector response R(Delta E_reco, E_true)

        Probability to reconstruct a given true energy in a given reconstructed
        energy band. In each reco bin, you integrate with a riemann sum over
        the default migra bin of your analysis.

        Parameters
        ----------
        e_true : `~astropy.units.Quantity`
            True energy
        e_reco : `~astropy.units.Quantity`, None
            Reconstructed energy axis
        offset : `~astropy.coordinates.Angle`
            Offset

        Returns
        -------
        rv : `~numpy.ndarray`
            Redistribution vector
        """
        e_true = Quantity(e_true)

        migra_axis = self.data.axis("migra")

        if e_reco is None:
            # Default: e_reco nodes = migra nodes * e_true nodes
            e_reco = migra_axis.edges * e_true
        else:
            # Translate given e_reco binning to migra at bin center
            e_reco = Quantity(e_reco)

        # migration value of e_reco bounds
        migra = e_reco / e_true

        values = self.data.evaluate(
            offset=offset, energy_true=e_true, migra=migra_axis.center
        )

        cumsum = np.insert(values, 0, 0).cumsum()

        with np.errstate(invalid="ignore"):
            cumsum = np.nan_to_num(cumsum / cumsum[-1])

        f = interp1d(
            migra_axis.edges.value,
            cumsum,
            kind="linear",
            bounds_error=False,
            fill_value=(0, 1),
        )

        # We compute the difference between 2 successive bounds in e_reco
        # to get integral over reco energy bin
        integral = np.diff(np.clip(f(migra), a_min=0, a_max=1))

        return integral

    def plot_migration(self, ax=None, offset=None, e_true=None, migra=None, **kwargs):
        """Plot energy dispersion for given offset and true energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`, optional
            Offset
        e_true : `~astropy.units.Quantity`, optional
            True energy
        migra : `~numpy.ndarray`, optional
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
            e_true = Quantity([0.1, 1, 10], "TeV")
        else:
            e_true = np.atleast_1d(Quantity(e_true))

        migra = self.data.axis("migra").center if migra is None else migra

        for ener in e_true:
            for off in offset:
                disp = self.data.evaluate(offset=off, energy_true=ener, migra=migra)
                label = f"offset = {off:.1f}\nenergy = {ener:.1f}"
                ax.plot(migra, disp, label=label, **kwargs)

        ax.set_xlabel(r"$E_\mathrm{{Reco}} / E_\mathrm{{True}}$")
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
            offset = Angle(1, "deg")

        e_true = self.data.axis("energy_true")
        migra = self.data.axis("migra")

        x = e_true.edges.value
        y = migra.edges.value

        z = self.data.evaluate(
            offset=offset,
            energy_true=e_true.center.reshape(1, -1, 1),
            migra=migra.center.reshape(1, 1, -1),
        ).value[0]

        caxes = ax.pcolormesh(x, y, z.T, **kwargs)

        if add_cbar:
            label = "Probability density (A.U.)"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        ax.set_xlabel(fr"$E_\mathrm{{True}}$ [{e_true.unit}]")
        ax.set_ylabel(r"$E_\mathrm{{Reco}} / E_\mathrm{{True}}$")
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

        energy = self.data.axis("energy_true").edges
        migra = self.data.axis("migra").edges
        theta = self.data.axis("offset").edges

        table = Table(meta=meta)
        table["ENERG_LO"] = energy[:-1][np.newaxis]
        table["ENERG_HI"] = energy[1:][np.newaxis]
        table["MIGRA_LO"] = migra[:-1][np.newaxis]
        table["MIGRA_HI"] = migra[1:][np.newaxis]
        table["THETA_LO"] = theta[:-1][np.newaxis]
        table["THETA_HI"] = theta[1:][np.newaxis]
        table["MATRIX"] = self.data.data.T[np.newaxis]
        return table

    def to_fits(self, name="ENERGY DISPERSION"):
        """Convert to `~astropy.io.fits.BinTable`."""
        return fits.BinTableHDU(self.to_table(), name=name)
