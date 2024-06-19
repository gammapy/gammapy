# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import scipy.special
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.units import Quantity
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from gammapy.maps import MapAxes, MapAxis, RegionGeom
from gammapy.utils.deprecation import deprecated_renamed_argument
from gammapy.visualization.utils import add_colorbar
from ..core import IRF

__all__ = ["EnergyDispersion2D"]


log = logging.getLogger(__name__)


class EnergyDispersion2D(IRF):
    """Offset-dependent energy dispersion matrix.

    Data format specification: :ref:`gadf:edisp_2d`

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        True energy axis.
    migra_axis : `MapAxis`
        Energy migration axis.
    offset_axis : `MapAxis`
        Field of view offset axis.
    data : `~numpy.ndarray`
        Energy dispersion probability density.

    Examples
    --------
    Read energy dispersion IRF from disk:

    >>> from gammapy.maps import MapAxis
    >>> from gammapy.irf import EnergyDispersion2D
    >>> filename = '$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz'
    >>> edisp2d = EnergyDispersion2D.read(filename, hdu="EDISP")

    Create energy dispersion matrix (`~gammapy.irf.EnergyDispersion`)
    for a given field of view offset and energy binning:

    >>> energy_axis = MapAxis.from_bounds(0.1, 20, nbin=60, unit="TeV", interp="log", name='energy')
    >>> edisp = edisp2d.to_edisp_kernel(offset='1.2 deg', energy_axis=energy_axis,
    ...                                 energy_axis_true=energy_axis.copy(name='energy_true'))

    See Also
    --------
    EnergyDispersion.
    """

    tag = "edisp_2d"
    required_axes = ["energy_true", "migra", "offset"]
    default_unit = u.one

    @property
    def _default_offset(self):
        if self.axes["offset"].nbin == 1:
            default_offset = self.axes["offset"].center
        else:
            default_offset = [1.0] * u.deg
        return default_offset

    def _mask_out_bounds(self, invalid):
        return (
            invalid[self.axes.index("energy_true")] & invalid[self.axes.index("migra")]
        ) | invalid[self.axes.index("offset")]

    @classmethod
    def from_gauss(
        cls, energy_axis_true, migra_axis, offset_axis, bias, sigma, pdf_threshold=1e-6
    ):
        """Create Gaussian energy dispersion matrix (`EnergyDispersion2D`).

        The output matrix will be Gaussian in (energy_true / energy).

        The ``bias`` and ``sigma`` should be either floats or arrays of same dimension than
        ``energy_true``. ``bias`` refers to the mean value of the ``migra``
        distribution minus one, i.e. ``bias=0`` means no bias.

        Note that, the output matrix is flat in offset.

        Parameters
        ----------
        energy_axis_true : `MapAxis`
            True energy axis.
        migra_axis : `~astropy.units.Quantity`
            Migra axis.
        offset_axis : `~astropy.units.Quantity`
            Bin edges of offset.
        bias : float or `~numpy.ndarray`
            Center of Gaussian energy dispersion, bias.
        sigma : float or `~numpy.ndarray`
            RMS width of Gaussian energy dispersion, resolution.
        pdf_threshold : float, optional
            Zero suppression threshold. Default is 1e-6.
        """
        axes = MapAxes([energy_axis_true, migra_axis, offset_axis])
        coords = axes.get_coord(mode="edges", axis_name="migra")

        migra_min = coords["migra"][:, :-1, :]
        migra_max = coords["migra"][:, 1:, :]

        # Analytical formula for integral of Gaussian
        s = np.sqrt(2) * sigma
        t1 = (migra_max - 1 - bias) / s
        t2 = (migra_min - 1 - bias) / s
        pdf = (scipy.special.erf(t1) - scipy.special.erf(t2)) / 2
        pdf = pdf / (migra_max - migra_min)

        # no offset dependence
        data = pdf.T * np.ones(axes.shape)
        data[data < pdf_threshold] = 0

        return cls(
            axes=axes,
            data=data.value,
        )

    @deprecated_renamed_argument(
        ["energy_true", "energy"],
        ["energy_axis_true", "energy_axis"],
        ["v1.3", "v1.3"],
        arg_in_kwargs=True,
    )
    def to_edisp_kernel(self, offset, energy_axis_true=None, energy_axis=None):
        """Detector response R(Delta E_reco, Delta E_true).

        Probability to reconstruct an energy in a given true energy band
        in a given reconstructed energy band.

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset.
        energy_axis_true : `~gammapy.maps.MapAxis`, optional
            True energy axis. Default is None.
        energy_axis : `~gammapy.maps.MapAxis`, optional
            Reconstructed energy axis. Default is None.

        Returns
        -------
        edisp : `~gammapy.irf.EDispKernel`
            Energy dispersion matrix.
        """
        from gammapy.makers.utils import make_edisp_kernel_map

        offset = Angle(offset)

        if isinstance(energy_axis, Quantity):
            energy_axis = MapAxis.from_energy_edges(energy_axis)
        if energy_axis is None:
            energy_axis = self.axes["energy_true"].copy(name="energy")

        if isinstance(energy_axis_true, Quantity):
            energy_axis_true = MapAxis.from_energy_edges(
                energy_axis_true,
                name="energy_true",
            )
        if energy_axis_true is None:
            energy_axis_true = self.axes["energy_true"]

        pointing = SkyCoord("0d", "0d")

        center = pointing.directional_offset_by(
            position_angle=0 * u.deg, separation=offset
        )
        geom = RegionGeom.create(region=center, axes=[energy_axis, energy_axis_true])

        edisp = make_edisp_kernel_map(geom=geom, edisp=self, pointing=pointing)
        return edisp.get_edisp_kernel()

    def normalize(self):
        """Normalise energy dispersion."""
        super().normalize(axis_name="migra")

    def plot_migration(self, ax=None, offset=None, energy_true=None, **kwargs):
        """Plot energy dispersion for given offset and true energy.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        offset : `~astropy.coordinates.Angle`, optional
            Offset. Default is None.
        energy_true : `~astropy.units.Quantity`, optional
            True energy. Default is None.
        **kwargs : dict
            Keyword arguments forwarded to `~matplotlib.pyplot.plot`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset = self._default_offset
        else:
            offset = np.atleast_1d(Angle(offset))

        if energy_true is None:
            energy_true = u.Quantity([0.1, 1, 10], "TeV")
        else:
            energy_true = np.atleast_1d(u.Quantity(energy_true))

        migra = self.axes["migra"]

        with quantity_support():
            for ener in energy_true:
                for off in offset:
                    disp = self.evaluate(
                        offset=off, energy_true=ener, migra=migra.center
                    )
                    label = f"offset = {off:.1f}\nenergy = {ener:.1f}"
                    ax.plot(migra.center, disp, label=label, **kwargs)

        migra.format_plot_xaxis(ax=ax)
        ax.set_ylabel("Probability density")
        ax.legend(loc="upper left")
        return ax

    def plot_bias(
        self,
        ax=None,
        offset=None,
        add_cbar=False,
        axes_loc=None,
        kwargs_colorbar=None,
        **kwargs,
    ):
        """Plot migration as a function of true energy for a given offset.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes. Default is None.
        offset : `~astropy.coordinates.Angle`, optional
            Offset. Default is None.
        add_cbar : bool, optional
            Add a colorbar to the plot. Default is False.
        axes_loc : dict, optional
            Keyword arguments passed to `~mpl_toolkits.axes_grid1.axes_divider.AxesDivider.append_axes`.
        kwargs_colorbar : dict, optional
            Keyword arguments passed to `~matplotlib.pyplot.colorbar`.
        kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.pcolormesh`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Matplotlib axes.
        """
        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("norm", PowerNorm(gamma=0.5))

        kwargs_colorbar = kwargs_colorbar or {}

        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset = self._default_offset

        energy_true = self.axes["energy_true"]
        migra = self.axes["migra"]

        z = self.evaluate(
            offset=offset,
            energy_true=energy_true.center.reshape(1, -1, 1),
            migra=migra.center.reshape(1, 1, -1),
        ).value[0]

        with quantity_support():
            caxes = ax.pcolormesh(energy_true.edges, migra.edges, z.T, **kwargs)

        energy_true.format_plot_xaxis(ax=ax)
        migra.format_plot_yaxis(ax=ax)

        if add_cbar:
            label = "Probability density [A.U]."
            kwargs_colorbar.setdefault("label", label)
            add_colorbar(caxes, ax=ax, axes_loc=axes_loc, **kwargs_colorbar)

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots.

        Parameters
        ----------
        figsize : tuple, optional
            Size of the resulting plot. Default is (15, 5).
        """
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.plot_bias(ax=axes[0])
        self.plot_migration(ax=axes[1])
        edisp = self.to_edisp_kernel(offset=self._default_offset[0])
        edisp.plot_matrix(ax=axes[2])

        plt.tight_layout()
