# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.special
from scipy.interpolate import interp1d
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from gammapy.maps import MapAxis
from gammapy.utils.nddata import NDDataArray
from gammapy.utils.scripts import make_path
from .edisp_kernel import EDispKernel

__all__ = ["EnergyDispersion2D"]


class EnergyDispersion2D:
    """Offset-dependent energy dispersion matrix.

    Data format specification: :ref:`gadf:edisp_2d`

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        True energy axis
    migra_axis : `MapAxis`
        Energy migration axis
    offset_axis : `MapAxis`
        Field of view offset axis
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
    >>> edisp = edisp2d.to_edisp_kernel(offset='1.2 deg', e_reco=energy, energy_true=energy)

    See Also
    --------
    EnergyDispersion
    """

    tag = "edisp_2d"
    default_interp_kwargs = dict(bounds_error=False, fill_value=None)
    """Default Interpolation kwargs for `~gammapy.utils.nddata.NDDataArray`. Extrapolate."""

    def __init__(
        self,
        energy_axis_true,
        migra_axis,
        offset_axis,
        data,
        interp_kwargs=None,
        meta=None,
    ):
        if interp_kwargs is None:
            interp_kwargs = self.default_interp_kwargs

        axes = [energy_axis_true, migra_axis, offset_axis]

        self.data = NDDataArray(axes=axes, data=data, interp_kwargs=interp_kwargs)
        self.meta = meta or {}

    def __str__(self):
        ss = self.__class__.__name__
        ss += f"\n{self.data}"
        return ss

    @classmethod
    def from_gauss(cls, energy_true, migra, bias, sigma, offset, pdf_threshold=1e-6):
        """Create Gaussian energy dispersion matrix (`EnergyDispersion2D`).

        The output matrix will be Gaussian in (energy_true / energy).

        The ``bias`` and ``sigma`` should be either floats or arrays of same dimension than
        ``energy_true``. ``bias`` refers to the mean value of the ``migra``
        distribution minus one, i.e. ``bias=0`` means no bias.

        Note that, the output matrix is flat in offset.

        Parameters
        ----------
        energy_true : `~astropy.units.Quantity`
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
        energy_true = Quantity(energy_true)
        # erf does not work with Quantities
        energy_axis_true = MapAxis.from_energy_edges(
            energy_true, interp="log", name="energy_true"
        )

        true2d, migra2d = np.meshgrid(energy_axis_true.center, migra)

        migra2d_lo = migra2d[:-1, :]
        migra2d_hi = migra2d[1:, :]

        # Analytical formula for integral of Gaussian
        s = np.sqrt(2) * sigma
        t1 = (migra2d_hi - 1 - bias) / s
        t2 = (migra2d_lo - 1 - bias) / s
        pdf = (scipy.special.erf(t1) - scipy.special.erf(t2)) / 2

        data = pdf.T[:, :, np.newaxis] * np.ones(len(offset) - 1)

        data[data < pdf_threshold] = 0

        offset_axis = MapAxis.from_edges(offset, name="offset")
        migra_axis = MapAxis.from_edges(migra, name="migra")
        return cls(
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            offset_axis=offset_axis,
            data=data,
        )

    @classmethod
    def from_table(cls, table):
        """Create from `~astropy.table.Table`."""
        # TODO: move this to MapAxis.from_table()

        if "ENERG_LO" in table.colnames:
            energy_axis_true = MapAxis.from_table(
                table, column_prefix="ENERG", format="gadf-dl3"
            )
        elif "ETRUE_LO" in table.colnames:
            energy_axis_true = MapAxis.from_table(
                table, column_prefix="ETRUE", format="gadf-dl3"
            )
        else:
            raise ValueError(
                'Invalid column names. Need "ENERG_LO/ENERG_HI" or "ETRUE_LO/ETRUE_HI"'
            )

        offset_axis = MapAxis.from_table(
            table, column_prefix="THETA", format="gadf-dl3"
        )
        migra_axis = MapAxis.from_table(table, column_prefix="MIGRA", format="gadf-dl3")

        matrix = table["MATRIX"].quantity[0].transpose()

        return cls(
            energy_axis_true=energy_axis_true,
            offset_axis=offset_axis,
            migra_axis=migra_axis,
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

    def to_edisp_kernel(self, offset, energy_true=None, energy=None):
        """Detector response R(Delta E_reco, Delta E_true)

        Probability to reconstruct an energy in a given true energy band
        in a given reconstructed energy band

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset
        energy_true : `~astropy.units.Quantity`, None
            True energy axis
        energy : `~astropy.units.Quantity`
            Reconstructed energy axis

        Returns
        -------
        edisp : `~gammapy.irf.EDispKernel`
            Energy dispersion matrix
        """
        offset = Angle(offset)

        # TODO: expect directly MapAxis here?
        if energy is None:
            energy_axis = self.data.axes["energy_true"].copy(name="energy")
        else:
            energy_axis = MapAxis.from_energy_edges(energy)

        if energy_true is None:
            energy_axis_true = self.data.axes["energy_true"]
        else:
            energy_axis_true = MapAxis.from_energy_edges(
                energy_true, name="energy_true"
            )

        data = []
        for value in energy_axis_true.center:
            vec = self.get_response(
                offset=offset, energy_true=value, energy=energy_axis.edges
            )
            data.append(vec)

        return EDispKernel(
            energy_axis=energy_axis,
            energy_axis_true=energy_axis_true,
            data=np.asarray(data),
        )

    def get_response(self, offset, energy_true, energy=None):
        """Detector response R(Delta E_reco, E_true)

        Probability to reconstruct a given true energy in a given reconstructed
        energy band. In each reco bin, you integrate with a riemann sum over
        the default migra bin of your analysis.

        Parameters
        ----------
        energy_true : `~astropy.units.Quantity`
            True energy
        energy : `~astropy.units.Quantity`, None
            Reconstructed energy axis
        offset : `~astropy.coordinates.Angle`
            Offset

        Returns
        -------
        rv : `~numpy.ndarray`
            Redistribution vector
        """
        energy_true = Quantity(energy_true)

        migra_axis = self.data.axes["migra"]

        if energy is None:
            # Default: energy nodes = migra nodes * energy_true nodes
            energy = migra_axis.edges * energy_true
        else:
            # Translate given energy binning to migra at bin center
            energy = Quantity(energy)

        # migration value of energy bounds
        migra = energy / energy_true

        values = self.data.evaluate(
            offset=offset, energy_true=energy_true, migra=migra_axis.center
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

        # We compute the difference between 2 successive bounds in energy
        # to get integral over reco energy bin
        integral = np.diff(np.clip(f(migra), a_min=0, a_max=1))

        return integral

    def plot_migration(
        self, ax=None, offset=None, energy_true=None, migra=None, **kwargs
    ):
        """Plot energy dispersion for given offset and true energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`, optional
            Offset
        energy_true : `~astropy.units.Quantity`, optional
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

        if energy_true is None:
            energy_true = Quantity([0.1, 1, 10], "TeV")
        else:
            energy_true = np.atleast_1d(Quantity(energy_true))

        migra = self.data.axes["migra"].center if migra is None else migra

        for ener in energy_true:
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

        energy_true = self.data.axes["energy_true"]
        migra = self.data.axes["migra"]

        x = energy_true.edges.value
        y = migra.edges.value

        z = self.data.evaluate(
            offset=offset,
            energy_true=energy_true.center.reshape(1, -1, 1),
            migra=migra.center.reshape(1, 1, -1),
        ).value[0]

        caxes = ax.pcolormesh(x, y, z.T, **kwargs)

        if add_cbar:
            label = "Probability density (A.U.)"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        ax.set_xlabel(fr"$E_\mathrm{{True}}$ [{energy_true.unit}]")
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
        edisp = self.to_edisp_kernel(offset="1 deg")
        edisp.plot_matrix(ax=axes[2])

        plt.tight_layout()

    def to_table(self):
        """Convert to `~astropy.table.Table`."""
        table = self.data.axes.to_table(format="gadf-dl3")
        table.meta = self.meta.copy()
        table["MATRIX"] = self.data.data.T[np.newaxis]
        return table

    def to_table_hdu(self, name="ENERGY DISPERSION"):
        """Convert to `~astropy.io.fits.BinTable`."""
        return fits.BinTableHDU(self.to_table(), name=name)
