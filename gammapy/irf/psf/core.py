# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
from astropy import units as u
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.array import array_stats_str
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from .table import EnergyDependentTablePSF, PSF3D
from ..core import IRF


class PSF(IRF):
    """"""
    def containment(self, rad):
        """"""
        pass

    def contaiment_radius(self, fraction):
        pass

    def info(
        self,
        fractions=[0.68, 0.95],
        energies=u.Quantity([1.0, 10.0], "TeV"),
        thetas=u.Quantity([0.0], "deg"),
    ):
        """
        Print PSF summary info.

        The containment radius for given fraction, energies and thetas is
        computed and printed on the command line.

        Parameters
        ----------
        fractions : list
            Containment fraction to compute containment radius for.
        energies : `~astropy.units.u.Quantity`
            Energies to compute containment radius for.
        thetas : `~astropy.units.u.Quantity`
            Thetas to compute containment radius for.

        Returns
        -------
        ss : string
            Formatted string containing the summary info.
        """
        ss = "\nSummary PSF info\n"
        ss += "----------------\n"
        ss += array_stats_str(self.axes["offset"].center.to("deg"), "Theta")
        ss += array_stats_str(self.axes["energy_true"].edges[1:], "Energy hi")
        ss += array_stats_str(self.axes["energy_true"].edges[:-1], "Energy lo")

        for fraction in fractions:
            containment = self.containment_radius(energies, thetas, fraction)
            for i, energy in enumerate(energies):
                for j, theta in enumerate(thetas):
                    radius = containment[j, i]
                    ss += (
                        "{:2.0f}% containment radius at theta = {} and "
                        "E = {:4.1f}: {:5.8f}\n"
                        "".format(100 * fraction, theta, energy, radius)
                    )
        return ss

    def plot_containment(self, fraction=0.68, ax=None, add_cbar=True, **kwargs):
        """
        Plot containment image with energy and theta axes.

        Parameters
        ----------
        fraction : float
            Containment fraction between 0 and 1.
        add_cbar : bool
            Add a colorbar
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.axes["energy_true"].center
        offset = self.axes["offset"].center

        # Set up and compute data
        containment = self.containment_radius(energy, offset, fraction)

        # plotting defaults
        kwargs.setdefault("cmap", "GnBu")
        kwargs.setdefault("vmin", np.nanmin(containment.value))
        kwargs.setdefault("vmax", np.nanmax(containment.value))

        # Plotting
        x = energy.value
        y = offset.value
        caxes = ax.pcolormesh(x, y, containment.value, **kwargs)

        # Axes labels and ticks, colobar
        ax.semilogx()
        ax.set_ylabel(f"Offset ({offset.unit})")
        ax.set_xlabel(f"Energy ({energy.unit})")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        if add_cbar:
            label = f"Containment radius R{100 * fraction:.0f} ({containment.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def plot_containment_vs_energy(
        self, fractions=[0.68, 0.95], thetas=[0, 1] * u.deg, ax=None, **kwargs
    ):
        """Plot containment fraction as a function of energy.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        energy = self.axes["energy_true"].center

        for theta in thetas:
            for fraction in fractions:
                radius = self.containment_radius(energy, theta, fraction).squeeze()
                kwargs.setdefault("label", f"{theta.to_value('deg')} deg, {100 * fraction:.1f}%")
                ax.plot(energy.value, radius.value, **kwargs)

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel("Energy (TeV)")
        ax.set_ylabel("Containment radius (deg)")

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


class ParametricPSF(PSF):
    """Parametric PSF base class

    Parameters
    -----------
    axes : list of `MapAxis` or `MapAxes`
        Axes
    data : dict of `~numpy.ndarray`, or `~numpy.recarray`
        Data
    unit : dict of str or `~astropy.units.Unit`
        Unit
    meta : dict
        Meta data
    """
    @property
    @abc.abstractmethod
    def required_parameters(self):
        pass

    @abc.abstractmethod
    def evaluate_direct(self, rad, ):
        pass

    @property
    def quantity(self):
        """Quantity"""
        quantity = {}

        for name in self.required_parameters:
            quantity[name] = self.data[name] * self.unit[name]

        return quantity

    @property
    def unit(self):
        """Map unit (`~astropy.units.Unit`)"""
        return self._unit

    @unit.setter
    def unit(self, values):
        self._unit = {key: u.Unit(val) for key, val in values.items()}

    @property
    def _interpolators(self):
        interps = {}

        for name in self.required_parameters:
            points = [a.center for a in self.axes]
            points_scale = tuple([a.interp for a in self.axes])
            interps[name] = ScaledRegularGridInterpolator(
                points, values=self.quantity[name], points_scale=points_scale
            )

        return interps

    def to_table(self, format="gadf-dl3"):
        """Convert PSF table data to table.

        Parameters
        ----------
        format : {"gadf-dl3"}
            Format specification


        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            PSF in HDU list format.
        """
        table = self.axes.to_table(format="gadf-dl3")

        for name in self.required_parameters:
            table[name.upper()] = self.data[name].T[np.newaxis]
            table[name.upper()].unit = self.unit[name]

        # Create hdu and hdu list
        return table

    @classmethod
    def from_table(cls, table, format="gadf-dl3"):
        """Create `PSFKing` from `~astropy.table.Table`.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Table King PSF info.
        """
        axes = MapAxes.from_table(table, format=format)[cls.required_axes]

        dtype = {"names": cls.required_parameters, "formats": len(cls.required_parameters) * (np.float32,)}

        data = np.empty(axes.shape, dtype=dtype)
        unit = {}

        for name in cls.required_parameters:
            column = table[name.upper()]
            values = column.data[0].transpose()

            # this fixes some files where sigma is written as zero
            if "SIGMA" in name:
                values[values == 0] = 1.

            data[name] = values.reshape(axes.shape)
            unit[name] = column.unit or ""

        return cls(
            axes=axes,
            data=data,
            meta=table.meta.copy(),
            unit=unit
        )

    def to_energy_dependent_table_psf(self, offset, rad=None):
        """Convert to energy-dependent table PSF.

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            Offset in the field of view. Default theta = 0 deg
        rad : `~astropy.coordinates.Angle`
            Offset from PSF center used for evaluating the PSF on a grid.
            Default offset = [0, 0.005, ..., 1.495, 1.5] deg.

        Returns
        -------
        table_psf : `~gammapy.irf.EnergyDependentTablePSF`
            Energy-dependent PSF
        """
        energy_axis_true = self.axes["energy_true"]

        if rad is None:
            rad = np.arange(0, 1.5, 0.005) * u.deg

        rad_axis = MapAxis.from_nodes(rad, name="rad")

        psf_value = u.Quantity(np.empty((energy_axis_true.nbin, len(rad))), "deg^-2")

        for idx, energy in enumerate(energy_axis_true.center):
            pars = self.evaluate(energy=energy, offset=offset)
            val = self.evaluate_direct(rad=rad, **pars)
            psf_value[idx] = u.Quantity(val, "deg^-2")

        return EnergyDependentTablePSF(
            axes=[energy_axis_true, rad_axis],
            data=psf_value.value,
            unit=psf_value.unit
        )

    def to_psf3d(self, rad=None):
        """Create a PSF3D from an analytical PSF.

        Parameters
        ----------
        rad : `~astropy.units.u.Quantity` or `~astropy.coordinates.Angle`
            the array of position errors (rad) on which the PSF3D will be defined

        Returns
        -------
        psf3d : `~gammapy.irf.PSF3D`
            the PSF3D. It will be defined on the same energy and offset values than the input psf.
        """
        offset_axis = self.axes["offset"]
        energy_axis_true = self.axes["energy_true"]

        if rad is None:
            rad = np.linspace(0, 0.66, 67) * u.deg

        rad_axis = MapAxis.from_edges(rad, name="rad")

        shape = (energy_axis_true.nbin, offset_axis.nbin, rad_axis.nbin)
        psf_value = np.zeros(shape) * u.Unit("sr-1")

        for idx, offset in enumerate(offset_axis.center):
            table_psf = self.to_energy_dependent_table_psf(offset)
            psf_value[:, idx, :] = table_psf.evaluate(
                energy_true=energy_axis_true.center[:, np.newaxis], rad=rad_axis.center
            )

        return PSF3D(
            axes=[energy_axis_true, offset_axis, rad_axis],
            data=psf_value.value,
            unit=psf_value.unit,
            meta=self.meta.copy()
        )
