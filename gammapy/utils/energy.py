# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.units import Quantity
from astropy.io import fits

__all__ = ["EnergyBounds"]

log = logging.getLogger(__name__)


def energy_logspace(emin, emax, nbins, unit=None, per_decade=False):
    """Create Energy with equal log-spacing (`~gammapy.utils.energy.Energy`).

    Parameters
    ----------
    emin : `~astropy.units.Quantity`, float
        Lowest energy bin
    emax : `~astropy.units.Quantity`, float
        Highest energy bin
    nbins : int
        Number of bins
    unit : `~astropy.units.UnitBase`, str
        Energy unit
    per_decade : bool
        Whether nbins is per decade.
    """
    if unit is not None:
        emin = Quantity(emin, unit)
        emax = Quantity(emax, unit)
    else:
        emin = Quantity(emin)
        emax = Quantity(emax)
        unit = emax.unit
        emin = emin.to(unit)

    x_min, x_max = np.log10([emin.value, emax.value])

    if per_decade:
        nbins = (x_max - x_min) * nbins

    energy = np.logspace(x_min, x_max, nbins)

    return Quantity(energy, unit, copy=False)


class EnergyBounds(Quantity):
    """EnergyBounds array.

    This is a `~gammapy.utils.energy.Energy` sub-class that adds convenience
    methods to handle common tasks for energy bin edges arrays, like FITS I/O or
    generating arrays of bin centers.

    See :ref:`energy_handling_gammapy` for further information.

    Parameters
    ----------
    energy : `~numpy.array`, scalar, `~astropy.units.Quantity`
        EnergyBounds
    unit : `~astropy.units.UnitBase`, str
        The unit of the values specified for the energy.  This may be any
        string that `~astropy.units.Unit` understands, but it is better to
        give an actual unit object.
    """

    @property
    def nbins(self):
        """The number of bins."""
        return self.size - 1

    @property
    def log_centers(self):
        """Log centers of the energy bounds."""
        center = np.sqrt(self[:-1] * self[1:])
        return center

    @property
    def upper_bounds(self):
        """Upper energy bin edges."""
        return self[1:]

    @property
    def lower_bounds(self):
        """Lower energy bin edges."""
        return self[:-1]

    @property
    def boundaries(self):
        """Energy range."""
        return self[[0, -1]]

    @property
    def bands(self):
        """Width of the energy bins."""
        upper = self.upper_bounds
        lower = self.lower_bounds
        return upper - lower

    @classmethod
    def from_lower_and_upper_bounds(cls, lower, upper, unit=None):
        """EnergyBounds from lower and upper bounds (`~gammapy.utils.energy.EnergyBounds`).

        If no unit is given, it will be taken from upper.

        Parameters
        ----------
        lower,upper : `~astropy.units.Quantity`, float
            Lowest and highest energy bin
        unit : `~astropy.units.UnitBase`, str, None
            Energy units
        """
        # np.append renders Quantities dimensionless
        # http://docs.astropy.org/en/latest/known_issues.html#quantity-issues

        if unit is None:
            unit = upper.unit
        lower = cls(lower, unit)
        upper = cls(upper, unit)
        energy = np.append(lower.value, upper.value[-1])
        return cls(energy, unit)

    @classmethod
    def equal_log_spacing(cls, emin, emax, nbins, unit=None):
        """EnergyBounds with equal log-spacing (`~gammapy.utils.energy.EnergyBounds`).

        If no unit is given, it will be taken from emax.

        Parameters
        ----------
        emin : `~astropy.units.Quantity`, float
            Lowest energy bin
        emax : `~astropy.units.Quantity`, float
            Highest energy bin
        bins : int
            Number of bins
        unit : `~astropy.units.UnitBase`, str, None
            Energy unit
        """
        return cls(energy_logspace(emin, emax, nbins + 1, unit))

    @classmethod
    def from_ebounds(cls, hdu):
        """Read EBOUNDS fits extension (`~gammapy.utils.energy.EnergyBounds`).

        Parameters
        ----------
        hdu: `~astropy.io.fits.BinTableHDU`
            ``EBOUNDS`` extensions.
        """
        if hdu.name != "EBOUNDS":
            log.warning(
                "This does not seem like an EBOUNDS extension. " "Are you sure?"
            )

        header = hdu.header
        unit = header.get("TUNIT2")
        low = hdu.data["E_MIN"]
        high = hdu.data["E_MAX"]
        return cls.from_lower_and_upper_bounds(low, high, unit)

    @classmethod
    def from_rmf_matrix(cls, hdu):
        """Read MATRIX fits extension (`~gammapy.utils.energy.EnergyBounds`).

        Parameters
        ----------
        hdu: `~astropy.io.fits.BinTableHDU`
            ``MATRIX`` extensions.
        """
        if hdu.name != "MATRIX":
            log.warning("This does not seem like a MATRIX extension. " "Are you sure?")

        header = hdu.header
        unit = header.get("TUNIT1")
        low = hdu.data["ENERG_LO"]
        high = hdu.data["ENERG_HI"]
        return cls.from_lower_and_upper_bounds(low, high, unit)

    def find_energy_bin(self, energy):
        """Find the bins that contain the specified energy values.

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`
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

        bin_index = np.searchsorted(self.upper_bounds, energy)

        return bin_index

    def contains(self, energy):
        """Check of energy is contained in boundaries.

        Parameters
        ----------
        energy : `~gammapy.utils.energy.Energy`
            Array of energies to test
        """
        return (energy > self[0]) & (energy < self[-1])

    def to_dict(self):
        """Construct dict representing an energy range."""
        if len(self) != 2:
            raise ValueError(
                "This is not an energy range. Nbins: {}".format(self.nbins)
            )

        d = dict(min=self[0].value, max=self[1].value, unit="{}".format(self.unit))

        return d

    @classmethod
    def from_dict(cls, d):
        """Read dict representing an energy range."""
        return cls((d["min"], d["max"]), d["unit"])
