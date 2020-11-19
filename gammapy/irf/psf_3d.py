# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.utils import lazyproperty
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from gammapy.utils.scripts import make_path
from .psf_table import EnergyDependentTablePSF, TablePSF

__all__ = ["PSF3D"]


class PSF3D:
    """PSF with axes: energy, offset, rad.

    Data format specification: :ref:`gadf:psf_table`

    Parameters
    ----------
    energy_axis_true : `MapAxis`
        True energy axis.
    offset_axis : `MapAxis`
        Offset axis
    rad_axis : `MapAxis`
        Rad axis
    psf_value : `~astropy.units.Quantity`
        PSF (3-dim with axes: psf[rad_index, offset_index, energy_index]
    energy_thresh_lo : `~astropy.units.Quantity`
        Lower energy threshold.
    energy_thresh_hi : `~astropy.units.Quantity`
        Upper energy threshold.
    """

    tag = "psf_table"

    def __init__(
        self,
        energy_axis_true,
        offset_axis,
        rad_axis,
        psf_value,
        energy_thresh_lo=u.Quantity(0.1, "TeV"),
        energy_thresh_hi=u.Quantity(100, "TeV"),
        interp_kwargs=None,
    ):
        if energy_axis_true.name != "energy_true":
            raise ValueError(
                "Unexpected `energy_axis_true.name`,"
                f' expected "energy_true", got: {energy_axis_true.name}'
            )

        if offset_axis.name != "offset":
            raise ValueError(
                "Unexpected `offset_axis.name`,"
                f' expected "offset", got: {offset_axis.name}'
            )
        if rad_axis.name != "rad":
            raise ValueError(
                "Unexpected `rad_axis.name`," f' expected "rad", got: {rad_axis.name}'
            )

        expected_shape = (energy_axis_true.nbin, offset_axis.nbin, rad_axis.nbin)
        if psf_value.shape != expected_shape:
            raise ValueError(
                "PSF has wrong shape"
                f", expected {expected_shape}, got {psf_value.shape}"
            )

        self._energy_axis_true = energy_axis_true
        self._offset_axis = offset_axis
        self._rad_axis = rad_axis
        self.psf_value = psf_value.to("sr^-1")
        self.energy_thresh_lo = energy_thresh_lo.to("TeV")
        self.energy_thresh_hi = energy_thresh_hi.to("TeV")

        self._interp_kwargs = interp_kwargs or {}

    @property
    def energy_axis_true(self):
        return self._energy_axis_true

    @property
    def rad_axis(self):
        return self._rad_axis

    @property
    def offset_axis(self):
        return self._offset_axis

    @lazyproperty
    def _interpolate(self):
        energy = self.energy_axis_true.center
        offset = self.offset_axis.center
        rad = self.rad_axis.center

        return ScaledRegularGridInterpolator(
            points=(energy, offset, rad), values=self.psf_value, **self._interp_kwargs
        )

    def __repr__(self):
        """Print some basic info.
        """
        info = self.__class__.__name__ + "\n"
        info += "-" * len(self.__class__.__name__) + "\n\n"
        info += f"\tshape      : {self.psf_value.shape}\n"
        return info

    @classmethod
    def read(cls, filename, hdu="PSF_2D_TABLE"):
        """Create `PSF3D` from FITS file.

        Parameters
        ----------
        filename : str
            File name
        hdu : str
            HDU name
        """
        table = Table.read(make_path(filename), hdu=hdu)
        return cls.from_table(table)

    @classmethod
    def from_table(cls, table):
        """Create `PSF3D` from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table Table-PSF info.
        """
        psf_value = table["RPSF"].quantity[0].transpose()

        opts = {}
        try:
            opts["energy_thresh_lo"] = u.Quantity(table.meta["LO_THRES"], "TeV")
            opts["energy_thresh_hi"] = u.Quantity(table.meta["HI_THRES"], "TeV")
        except KeyError:
            pass

        energy_axis_true = MapAxis.from_table(
            table, column_prefix="ENERG", format="gadf-dl3"
        )
        offset_axis = MapAxis.from_table(
            table, column_prefix="THETA", format="gadf-dl3"
        )
        rad_axis = MapAxis.from_table(table, column_prefix="RAD", format="gadf-dl3")

        return cls(
            energy_axis_true=energy_axis_true,
            offset_axis=offset_axis,
            rad_axis=rad_axis,
            psf_value=psf_value,
            **opts,
        )

    def to_hdulist(self):
        """Convert PSF table data to FITS HDU list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        axes = MapAxes([self.offset_axis, self.energy_axis_true, self.rad_axis])
        table = axes.to_table(format="gadf-dl3")

        table["RPSF"] = self.psf_value.T[np.newaxis]

        hdu = fits.BinTableHDU(table)
        hdu.header["LO_THRES"] = self.energy_thresh_lo.value
        hdu.header["HI_THRES"] = self.energy_thresh_hi.value

        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def write(self, filename, *args, **kwargs):
        """Write PSF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_hdulist().writeto(str(make_path(filename)), *args, **kwargs)

    def evaluate(self, energy=None, offset=None, rad=None):
        """Interpolate PSF value at a given offset and energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            energy value
        offset : `~astropy.coordinates.Angle`
            Offset in the field of view
        rad : `~astropy.coordinates.Angle`
            Offset wrt source position

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        if energy is None:
            energy = self.energy_axis_true.center
        if offset is None:
            offset = self.offset_axis.center
        if rad is None:
            rad = self.rad_axis.center

        rad = np.atleast_1d(u.Quantity(rad))
        offset = np.atleast_1d(u.Quantity(offset))
        energy = np.atleast_1d(u.Quantity(energy))
        return self._interpolate(
            (
                energy[np.newaxis, np.newaxis, :],
                offset[np.newaxis, :, np.newaxis],
                rad[:, np.newaxis, np.newaxis],
            )
        )

    def to_energy_dependent_table_psf(self, theta="0 deg", rad=None, exposure=None):
        """
        Convert PSF3D in EnergyDependentTablePSF.

        Parameters
        ----------
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid.
            Default is the ``rad`` from this PSF.
        exposure : `~astropy.units.Quantity`
            Energy dependent exposure. Should be in units equivalent to 'cm^2 s'.
            Default exposure = 1.

        Returns
        -------
        table_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Energy-dependent PSF
        """
        theta = Angle(theta)

        if rad is not None:
            rad_axis = MapAxis.from_edges(rad, name="rad")
        else:
            rad_axis = self.rad_axis

        psf_value = self.evaluate(offset=theta, rad=rad_axis.center).squeeze()
        return EnergyDependentTablePSF(
            energy_axis_true=self.energy_axis_true,
            rad_axis=rad_axis,
            exposure=exposure,
            psf_value=psf_value.transpose(),
        )

    def to_table_psf(self, energy, theta="0 deg", **kwargs):
        """Create `~gammapy.irf.TablePSF` at one given energy.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg

        Returns
        -------
        psf : `~gammapy.irf.TablePSF`
            Table PSF
        """
        energy = u.Quantity(energy)
        theta = Angle(theta)
        psf_value = self.evaluate(energy, theta).squeeze()
        return TablePSF(rad_axis=self.rad_axis, psf_value=psf_value, **kwargs)

    def containment_radius(
        self, energy, theta="0 deg", fraction=0.68, interp_kwargs=None
    ):
        """Containment radius.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        fraction : float
            Containment fraction. Default fraction = 0.68

        Returns
        -------
        radius : `~astropy.units.Quantity`
            Containment radius in deg
        """
        energy = np.atleast_1d(u.Quantity(energy))
        theta = np.atleast_1d(u.Quantity(theta))

        radii = []
        for t in theta:
            psf = self.to_energy_dependent_table_psf(theta=t)
            radii.append(psf.containment_radius(energy, fraction=fraction))

        return u.Quantity(radii).T.squeeze()

    def plot_containment_vs_energy(
        self, fractions=[0.68, 0.95], thetas=Angle([0, 1], "deg"), ax=None, **kwargs
    ):
        """Plot containment fraction as a function of energy.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = MapAxis.from_energy_bounds(
            self.energy_axis_true.edges[0], self.energy_axis_true.edges[-1], 100
        ).edges

        for theta in thetas:
            for fraction in fractions:
                plot_kwargs = kwargs.copy()
                radius = self.containment_radius(energy, theta, fraction)
                plot_kwargs.setdefault(
                    "label", f"{theta.deg} deg, {100 * fraction:.1f}%"
                )
                ax.plot(energy.value, radius.value, **plot_kwargs)

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel("Energy (TeV)")
        ax.set_ylabel("Containment radius (deg)")

    def plot_psf_vs_rad(self, theta="0 deg", energy=u.Quantity(1, "TeV")):
        """Plot PSF vs rad.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy. Default energy = 1 TeV
        theta : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        """
        theta = Angle(theta)
        table = self.to_table_psf(energy=energy, theta=theta)
        return table.plot_psf_vs_rad()

    def plot_containment(self, fraction=0.68, ax=None, add_cbar=True, **kwargs):
        """Plot containment image with energy and theta axes.

        Parameters
        ----------
        fraction : float
            Containment fraction between 0 and 1.
        add_cbar : bool
            Add a colorbar
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.energy_axis_true.center
        offset = self.offset_axis.center

        # Set up and compute data
        containment = self.containment_radius(energy, offset, fraction)

        # plotting defaults
        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("vmin", np.nanmin(containment.value))
        kwargs.setdefault("vmax", np.nanmax(containment.value))

        # Plotting
        x = energy.value
        y = offset.value
        caxes = ax.pcolormesh(x, y, containment.value.T, **kwargs)

        # Axes labels and ticks, colobar
        ax.semilogx()
        ax.set_ylabel(f"Offset ({offset.unit})")
        ax.set_xlabel(f"Energy ({energy.unit})")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        try:
            self._plot_safe_energy_range(ax)
        except KeyError:
            pass

        if add_cbar:
            label = f"Containment radius R{100 * fraction:.0f} ({containment.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def _plot_safe_energy_range(self, ax):
        """add safe energy range lines to the plot"""
        esafe = self.energy_thresh_lo
        omin = self.offset_axis.center.value.min()
        omax = self.offset_axis.center.value.max()
        ax.vlines(x=esafe.value, ymin=omin, ymax=omax)
        label = f"Safe energy threshold: {esafe:3.2f}"
        ax.text(x=0.1, y=0.9 * esafe.value, s=label, va="top")

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        self.plot_containment(fraction=0.68, ax=axes[0])
        self.plot_containment(fraction=0.95, ax=axes[1])
        self.plot_containment_vs_energy(ax=axes[2])

        # TODO: implement this plot
        # psf = self.psf_at_energy_and_theta(energy='1 TeV', theta='1 deg')
        # psf.plot_components(ax=axes[2])

        plt.tight_layout()
