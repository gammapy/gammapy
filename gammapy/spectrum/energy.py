# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.units import Quantity
from astropy.io import fits
from ..utils.array import array_stats_str
from astropy import log

import astropy.units as u
import numpy as np

__all__ = ['Energy',
           'EnergyBounds']

class Energy(u.Quantity):
    """Energy quantity scalar or array.

    This is a `~astropy.units.Quantity` sub-class that adds convenience methods
    to handle common tasks for energy bin center arrays, like FITS I/O or generating
    equal-log-spaced grids of energies.

    See :ref:`energy_handling_gammapy` for further information.


    Parameters
    ----------
    energy : `~numpy.array`, scalar, `~astropy.units.Quantity`
        Energy
    unit : `~astropy.units.UnitBase`, str, optional
        The unit of the value specified for the energy.  This may be
        any string that `~astropy.units.Unit` understands, but it is
        better to give an actual unit object.
    dtype : `~numpy.dtype`, optional
        See `~astropy.units.Quantity`.
    copy : bool, optional
        See `~astropy.units.Quantity`.

    """

    def __new__(cls, energy, unit=None, dtype=None, copy=True):

        # Techniques to subclass Quantity taken from astropy.coordinates.Angle
        # see: http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

        self = super(Energy, cls).__new__(cls, energy, unit,
                                          dtype=dtype, copy=copy)

        if not self.unit.is_equivalent(u.eV):
            raise ValueError("Given unit {0} is not an"
                             " energy".format(self.unit.to_string()))

        return self

    def __array_finalize__(self, obj):
        super(Energy, self).__array_finalize__(obj)

    def __quantity_subclass__(self, unit):
        if unit.is_equivalent(u.eV):
            return Energy, True
        else:
            return super(Energy, self).__quantity_subclass__(unit)[0], False

    @property
    def nbins(self):
        """
        The number of bins
        """
        return self.size

    @classmethod
    def equal_log_spacing(cls, emin, emax, nbins, unit=None):
        """Create Energy with equal log-spacing (`~gammapy.spectrum.energy.Energy`).

        if no unit is given, it will be taken from emax

        Parameters
        ----------
        emin : `~astropy.units.Quantity`, float
            Lowest energy bin
        emax : `~astropy.units.Quantity`, float
            Highest energy bin
        bins : int
            Number of bins
        unit : `~astropy.units.UnitBase`, str
            Energy unit
        """

        if unit is None:
            unit = emax.unit

        emin = Energy(emin, unit)
        emax = Energy(emax, unit)

        x_min, x_max = np.log10([emin.value, emax.value])
        energy = np.logspace(x_min, x_max, nbins)

        return cls(energy, unit, copy=False)

    @classmethod
    def from_fits(cls, hdu, unit=None):
        """Read ENERGIES fits extension (`~gammapy.spectrum.energy.Energy`).

        Parameters
        ----------
        hdu: `~astropy.io.fits.BinTableHDU`
            ``ENERGIES`` extensions.
        unit : `~astropy.units.UnitBase`, str, None
            Energy unit
        """

        header = hdu.header
        fitsunit = header.get('TUNIT1')

        if fitsunit is None:
            if unit is not None:
                log.warn("No unit found in the FITS header."
                         " Setting it to {0}".format(unit))
                fitsunit = unit
            else:
                raise ValueError("No unit found in the FITS header."
                                 " Please specifiy a unit")

        energy = cls(hdu.data['Energy'], fitsunit)

        return energy.to(unit)
        

    def to_fits(self, **kwargs):
        """Write ENERGIES fits extension

        Returns
        -------
        hdu: `~astropy.io.fits.BinTableHDU`
            ENERGIES fits extension
        """

        col1 = fits.Column(name='Energy', format='D', array=self.value)
        cols = fits.ColDefs([col1])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.name = 'ENERGIES'
        hdu.header['TUNIT1'] = "{0}".format(self.unit.to_string('fits'))

        return hdu


class EnergyBounds(Energy):

    """EnergyBounds array.

    This is a `~gammapy.spectrum.energy.Energy` sub-class that adds convenience 
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
        """
        The number of bins
        """
        return self.size - 1

    @property
    def log_centers(self):
        """Log centers of the energy bounds
        """
        
        center = np.sqrt(self[:-1] * self[1:])
        return center.view(Energy)

    @property
    def upper_bounds(self):
        """Upper energy bin edges
        """
        return self[1:]

    @property
    def lower_bounds(self):
        """Lower energy bin edges
        """
        
        return self[:-1]

    @property
    def bands(self):
        """Width of the energy bins
        """
        
        upper = self.upper_bounds
        lower = self.lower_bounds
        return upper-lower


    @classmethod
    def from_lower_and_upper_bounds(cls, lower, upper, unit=None):
        """EnergyBounds from lower and upper bounds (`~gammapy.spectrum.energy.EnergyBounds`). 

        If no unit is given, it will be taken from upper
        
        Parameters
        ----------
        lower,upper : `~astropy.units.Quantity`, float
            Lowest and highest energy bin
        unit : `~astropy.units.UnitBase`, str, None
            Energy units
        """
        
        #np.append renders Quantities dimensionless
        #http://astropy.readthedocs.org/en/latest/known_issues.html#quantity-issues
        
        lower = cls(lower, unit);
        upper = cls(upper, unit);
        unit = upper.unit
        energy = np.append(lower, upper[-1])
        return cls(energy.value, unit)
        
    @classmethod
    def equal_log_spacing(cls, emin, emax, nbins, unit=None):
        """EnergyBounds with equal log-spacing (`~gammapy.spectrum.energy.EnergyBounds`).

        If no unit is given, it will be taken from emax

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

        return super(EnergyBounds, cls).equal_log_spacing(
            emin, emax, nbins + 1, unit)

    @classmethod
    def from_fits(cls, hdu, unit=None):
        """Read EBOUNDS fits extension (`~gammapy.spectrum.energy.EnergyBounds`).

        Parameters
        ----------
        hdu: `~astropy.io.fits.BinTableHDU`
            ``EBOUNDS`` extensions.
        unit : `~astropy.units.UnitBase`, str, None
            Energy unit
        """
        
        if hdu.name != 'EBOUNDS':
            log.warn('This does not seem like an EBOUNDS extension. Are you sure?')

        return super(EnergyBounds, cls).from_fits(cls, hdu, unit)

    def to_fits(self, **kwargs):
        """Write EBOUNDS fits extension

        Returns
        -------
        hdu: `~astropy.io.fits.BinTableHDU`
            EBOUNDS fits extension
        """

        col1 = fits.Column(name='Energy', format='D', array=self.value)
        cols = fits.ColDefs([col1])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.name = 'EBOUNDS'
        hdu.header['TUNIT1'] = "{0}".format(self.unit.to_str('fits'))

        return hdu
