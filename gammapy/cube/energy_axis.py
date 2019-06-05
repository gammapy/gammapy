# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import astropy.units as u
from astropy.io import fits
from ..maps import MapAxis

__all__ = ["EnergyAxis"]

class EnergyAxis(MapAxis):
    """A specialized MapAxis for energy

    This can be used both for true or reco energy.

    Parameters
    ----------
    nodes : `~astropy.units.Quantity`
        a vector of energies
    interp : str, `lin`or `log`
        the interpolation mode, default 'log'
    node_edges : str
        type of axis: 'edges' or 'center'
    is_e_true : bool
        true energy axis if True, reco energy otherwise , Default False
    """

    def __init__(self, nodes, unit=None, interp="log", node_type="edges", is_e_true=False):
        if unit is None:
            unit = nodes.unit
            nodes = nodes.value

        super().__init__(nodes=nodes, interp=interp, name="energy",
                         node_type=node_type, unit=unit)

        if not self.unit.is_equivalent("eV"):
            raise ValueError(
                "Given unit {} is not an" " energy".format(self.unit.to_string())
            )

        if is_e_true == True:
            self.e_true = True
        else:
            self.e_true = False

    @property
    def energy_lo(self):
        return self.edges[:-1]

    @property
    def energy_hi(self):
        return self.edges[1:]

    @property
    def energy(self):
        return self.edges

    @property
    def boundaries(self):
        """Energy range."""
        return u.Quantity([self.edges[0].value, self.edges[-1].value], self.unit)

    @property
    def bands(self):
        """Width of the energy bins."""
        upper = self.energy_hi
        lower = self.energy_lo
        return upper - lower

    @classmethod
    def from_lower_and_upper_bounds(cls, lower, upper, unit=None, interp='log', is_e_true=False):
        """EnergyAxis from lower and upper bounds.

        If no unit is given, it will be taken from upper.

        Parameters
        ----------
        lower,upper : `~astropy.units.Quantity`, float
            Lowest and highest energy bin
        unit : `~astropy.units.UnitBase`, str, None
            Energy units
        interp : str, `lin`or `log`
            the interpolation mode, default 'log'
        is_e_true : bool
            true energy axis if True, reco energy otherwise , Default False
        """
        # np.append renders Quantities dimensionless
        # http://docs.astropy.org/en/latest/known_issues.html#quantity-issues

        if unit is None:
            unit = upper.unit
        lower = u.Quantity(lower, unit)
        upper = u.Quantity(upper, unit)
        energy = u.Quantity(np.append(lower.value, upper.value[-1]), unit)
        return cls(energy, interp=interp, node_type='edges', is_e_true=is_e_true)

    @classmethod
    def equal_log_spacing(cls, emin, emax, nbins, unit=None, node_type='edges', is_e_true=False):
        """EnergyAxis with equal log-spacing (`~gammapy.utils.energy.EnergyBounds`).

        If no unit is given, it will be taken from emax.

        Parameters
        ----------
        emin : `~astropy.units.Quantity`, float
            Lowest energy bin
        emax : `~astropy.units.Quantity`, float
            Highest energy bin
        nbins : int
            Number of bins
        unit : `~astropy.units.UnitBase`, str, None
            Energy unit
        node_type : str
            type of axis: 'edges' or 'center'
        is_e_true : bool
            true energy axis if True, reco energy otherwise , Default False
        """
        if unit is not None:
            emin = u.Quantity(emin, unit)
            emax = u.Quantity(emax, unit)
        else:
            emin = u.Quantity(emin)
            emax = u.Quantity(emax)
            unit = emax.unit
            emin = emin.to(unit)

        x_min, x_max = np.log10([emin.value, emax.value])
        if node_type == 'edges':
            energy = u.Quantity(np.logspace(x_min, x_max, nbins + 1), unit)
        else:
            energy = u.Quantity(np.logspace(x_min, x_max, nbins ), unit)

        return cls(nodes=energy, interp='log', node_type=node_type)

    def find_energy_bin(self, energy):
        """Find the bins that contain the specified energy values.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Array of energies to search for.

        Returns
        -------
        bin_index : `~numpy.ndarray`
            Indices of the energy bins containing the specified energies.
        """
        # check that the specified energy is within the boundaries
        if not self.contains(energy).all():
            ss_error = "Specified energy {}".format(energy)
            ss_error += " is outside the boundaries {}".format(self.boundaries)
            raise ValueError(ss_error)

        bin_index = np.searchsorted(self.energy_hi, energy)

        return bin_index

    def contains(self, energy):
        """Check if energy is contained in boundaries.

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`
            Array of energies to test
        """
        # TODO: Check if center_nodes should be working this way.
        return (energy >= self.edges[0]) & (energy <= self.edges[-1])

