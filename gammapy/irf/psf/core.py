# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np
from astropy import units as u
from gammapy.maps import MapAxes
from gammapy.utils.array import array_stats_str
from gammapy.utils.interpolation import ScaledRegularGridInterpolator

from ..core import IRF


class ParametricPSF(IRF):
    """Parametric PSF base class"""
    @property
    @abc.abstractmethod
    def par_names(self):
        pass

    @property
    @abc.abstractmethod
    def par_units(self):
        pass

    @property
    def _interpolators(self):
        interps = {}

        for name in self.par_names:
            points = [a.center for a in self.axes]
            points_scale = tuple([a.interp for a in self.axes])
            interps[name] = ScaledRegularGridInterpolator(
                points, values=self.data[name], points_scale=points_scale
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

        for name, unit in zip(self.par_names, self.par_units):
            table[name.upper()] = self.data[name].T[np.newaxis]
            table[name.upper()].unit = unit

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

        dtype = {"names": cls.par_names, "formats": len(cls.par_names) * (np.float32,)}

        data = np.empty(axes.shape, dtype=dtype)

        for name in cls.par_names:
            values = table[name.upper()].data[0].transpose()

            # this fixes some files where sigma is written as zero
            if "SIGMA" in name:
                values[values == 0] = 1.

            data[name] = values.reshape(axes.shape)

        return cls(
            axes=axes,
            data=data,
            meta=table.meta.copy(),
        )

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
