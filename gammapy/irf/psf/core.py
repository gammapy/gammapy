# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
from astropy import units as u
from gammapy.maps import MapAxes, MapAxis
from gammapy.utils.array import array_stats_str
from gammapy.utils.interpolation import ScaledRegularGridInterpolator
from ..core import IRF


class PSF(IRF):
    """PSF base class"""

    def normalize(self):
        """Normalize PSF to integrate to unity"""
        rad_max = self.axes["rad"].edges.max()
        self.data /= self.containment(rad=rad_max)

    def containment(self, rad, **kwargs):
        """Containment tof the PSF at given axes coordinates

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad value
        **kwargs : dict
            Other coordinates

        Returns
        -------
        containment : `~numpy.ndarray`
            Containment
        """
        containment = self.integral(axis_name="rad", rad=rad, **kwargs)
        return np.clip(containment.to(""), 0, 1)

    def containment_radius(self, fraction, factor=20, **kwargs):
        """Containment radius at given axes coordinates

        Parameters
        ----------
        fraction : float or `~numpy.ndarray`
            Containment fraction
        factor : int
            Up-sampling factor of the rad axis, determines the precision of the
            computed containment radius.
        **kwargs : dict
            Other coordinates

        Returns
        -------
        radius : `~astropy.coordinates.Angle`
            Containment radius
        """
        from gammapy.datasets.map import RAD_AXIS_DEFAULT
        # TODO: this uses a lot of numpy broadcasting, maybe simplify
        output = np.broadcast(*kwargs.values(), fraction)

        try:
            rad_axis = self.axes["rad"]
        except KeyError:
            rad_axis = RAD_AXIS_DEFAULT

        # upsample for better precision
        rad = rad_axis.upsample(factor=factor).center

        axis = tuple(range(output.ndim))
        rad = np.expand_dims(rad, axis=axis).T
        containment = self.containment(rad=rad, **kwargs)

        fraction_idx = np.argmin(np.abs(containment - fraction), axis=0)
        return rad[fraction_idx].reshape(output.shape)

    def info(
        self,
        fraction=[0.68, 0.95],
        energy_true=[[1.0], [10.0]] * u.TeV,
        offset=0*u.deg,
    ):
        """
        Print PSF summary info.

        The containment radius for given fraction, energies and thetas is
        computed and printed on the command line.

        Parameters
        ----------
        fraction : list
            Containment fraction to compute containment radius for.
        energy_true : `~astropy.units.u.Quantity`
            Energies to compute containment radius for.
        offset : `~astropy.units.u.Quantity`
            Offset to compute containment radius for.

        Returns
        -------
        ss : string
            Formatted string containing the summary info.
        """
        info = "\nSummary PSF info\n"
        info += "----------------\n"
        info += array_stats_str(self.axes["offset"].center.to("deg"), "Theta")
        info += array_stats_str(self.axes["energy_true"].edges[1:], "Energy hi")
        info += array_stats_str(self.axes["energy_true"].edges[:-1], "Energy lo")

        containment_radius = self.containment_radius(
            energy_true=energy_true, offset=offset, fraction=fraction
        )

        energy_true, offset, fraction = np.broadcast_arrays(
            energy_true, offset, fraction, subok=True
        )

        for idx in np.ndindex(containment_radius.shape):
            info += f"{100 * fraction[idx]:.2f} containment radius "
            info += f"at offset = {offset[idx]} "
            info += f"and energy_true = {energy_true[idx]:4.1f}: "
            info += f"{containment_radius[idx]:.3f}\n"

        return info

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
                plot_kwargs = kwargs.copy()
                radius = self.containment_radius(
                    energy_true=energy, offset=theta, fraction=fraction
                )
                plot_kwargs.setdefault(
                    "label", f"{theta}, {100 * fraction:.1f}%"
                )
                ax.plot(energy.value, radius.value, **plot_kwargs)

        ax.semilogx()
        ax.legend(loc="best")
        ax.set_xlabel("Energy (TeV)")
        ax.set_ylabel("Containment radius (deg)")

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

        energy = self.axes["energy_true"].center
        offset = self.axes["offset"].center

        # Set up and compute data
        containment = self.containment_radius(
            energy_true=energy[:, np.newaxis], offset=offset, fraction=fraction
        )

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

        if add_cbar:
            label = f"Containment radius R{100 * fraction:.0f} ({containment.unit})"
            ax.figure.colorbar(caxes, ax=ax, label=label)

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

        self.plot_containment(fraction=0.68, ax=axes[0])
        self.plot_containment(fraction=0.95, ax=axes[1])
        self.plot_containment_vs_energy(ax=axes[2])
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

    def evaluate_parameters(self, energy_true, offset):
        """Evaluate analytic PSF parameters at a given energy and offset.

        Uses nearest-neighbor interpolation.

        Parameters
        ----------
        energy_true : `~astropy.units.Quantity`
            energy value
        offset : `~astropy.coordinates.Angle`
            Offset in the field of view

        Returns
        -------
        values : `~astropy.units.Quantity`
            Interpolated value
        """
        pars = {}
        for name in self.required_parameters:
            value = self._interpolators[name]((energy_true, offset))
            pars[name] = value

        return pars

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

        dtype = {
            "names": cls.required_parameters,
            "formats": len(cls.required_parameters) * (np.float32,)
        }

        data = np.empty(axes.shape, dtype=dtype)
        unit = {}

        for name in cls.required_parameters:
            column = table[name.upper()]
            values = column.data[0].transpose()

            # this fixes some files where sigma is written as zero
            if "SIGMA" in name:
                values[values == 0] = 1.

            # this reshape relies on a correct convention
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
        from gammapy.irf import EnergyDependentTablePSF

        energy_axis_true = self.axes["energy_true"]

        if rad is None:
            rad = np.arange(0, 1.5, 0.005) * u.deg

        rad_axis = MapAxis.from_nodes(rad, name="rad")

        psf_value = u.Quantity(np.empty((energy_axis_true.nbin, len(rad))), "deg^-2")

        for idx, energy in enumerate(energy_axis_true.center):
            val = self.evaluate(rad=rad, energy_true=energy, offset=offset)
            psf_value[idx] = u.Quantity(val, "deg^-2")

        return EnergyDependentTablePSF(
            axes=[energy_axis_true, rad_axis],
            data=psf_value.value,
            unit=psf_value.unit
        )

    def to_psf3d(self, rad=None):
        """Create a PSF3D from an analytical PSF.

        It will be defined on the same energy and offset values than the input psf.

        Parameters
        ----------
        rad : `~astropy.units.Quantity`
            Rad values

        Returns
        -------
        psf3d : `~gammapy.irf.PSF3D`
            PSF3D.
        """
        from gammapy.irf import PSF3D
        from gammapy.datasets.map import RAD_AXIS_DEFAULT

        offset_axis = self.axes["offset"]
        energy_axis_true = self.axes["energy_true"]

        if rad is None:
            rad_axis = RAD_AXIS_DEFAULT.center
        else:
            rad_axis = MapAxis.from_edges(rad, name="rad")

        axes = MapAxes([energy_axis_true, offset_axis, rad_axis])
        psf_value = self.evaluate(**axes.get_coord())

        return PSF3D(
            axes=[energy_axis_true, offset_axis, rad_axis],
            data=psf_value.value,
            unit=psf_value.unit,
            meta=self.meta.copy()
        )
