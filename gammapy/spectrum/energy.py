# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.units import Quantity
from ..utils.array import array_stats_str

import astropy.units as u
import numpy as np

__all__ = ['Energy',
           'EnergyBounds']


class Energy(u.Quantity):

    """Energy bin centers.

    Stored as "ENERGIES" FITS table extensions.

    Parameters
    ----------
    energy : `~numpy.array`, scalar, `~astropy.units.Quantity`,
                        :class:`~gammapy.spectrum.energy.Energy`
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
        
    #Techniques to subclass Quantity taken from astropy.coordinates.Angle
    #see also: http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        if unit is not None:
            unit = u.Unit(unit)
            if not unit.is_equivalent(u.eV):
                raise ValueError("Requested unit {0} is not an"
                                 " energy unit".format(unit)) 

        if isinstance(energy, u.Quantity):
            if unit is not None:
                energy = energy.to(unit).value
            else:
                unit = energy.unit
                if not unit.is_equivalent(u.eV):
                    raise ValueError("Given quantity {0} is not an"
                                     " energy".format(energy))
                energy = energy.value

        else:
            if unit is None:
                raise ValueError("No unit given")
            
        self = super(Energy, cls).__new__(cls, energy, unit, dtype=dtype, copy=copy)

        return self

    @staticmethod
    def equal_log_spacing(emin, emax, nbins, unit=None):
        """Create Energy with equal log-spacing.

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
        
        Returns
        -------
        Energy
        """

        if unit is None:
            unit = emax.unit
            
        emin = Quantity(emin, unit)
        emax = Quantity(emax, unit)

        x_min, x_max = np.log10([emin.value, emax.value])
        energy = np.logspace(x_min, x_max, nbins)
        energy = Quantity(energy, unit)

        return Energy(energy)

    @staticmethod
    def from_fits(hdu):
        """Read ENERGIES fits extension.

        Parameters
        ---------
        hdu: `~astropy.io.fits.BinTableHDU`
            ``ENERGIES`` extensions.

        Returns
        ------
        Energy
        """
        
        header = hdu.header
        energy = Quantity(hdu.data['Energy'], header['TUNIT1'])
        return Energy(energy)

    def to_fits(self, **kwargs):
        """Write ENERGIES fits extension

        Returns
        -------
        hdu: `~astropy.io.fits.BinTableHDU`
            ENERGIES fits extension


        """

        col1 = fits.Column(name='Energy', format='D', array=self.value)
        cols = fits.ColDefs([col1])
        return fits.BinTableHDU.from_columns(cols)

class EnergyBounds(Quantity):

    """Energy bin edges

    Stored as "EBOUNDS" FITS table extensions.

    Parameters
    ----------

    energy : `~numpy.ndarray`
        Energy
    unit : `~astropy.units.UnitBase`, str
        The unit of the values specified for the energy.  This may be any
        string that `~astropy.units.Unit` understands, but it is better to
        give an actual unit object.

    """

    def __init__(self, energy, unit=None):

        if isinstance(energy, Quantity):
            energy = energy.value
            unit = energy.unit

        if isinstance(unit, u.UnitBase):
            pass
        elif isinstance(unit, basestring):
            unit = u.Unit(unit)
        elif unit is None:
            raise UnitsError("No unit was specified in EnergyBounds initializer")

        self._unit = unit
        self._value = energy

    @staticmethod
    def equal_log_spacing(emin, emax, nbins):
        """Create EnergyBounds with equal log-spacing.

        The unit will be taken from emax

        Parameters
        ----------
        emin : `~astropy.units.Quantity`
            Lowest energy bin
        emax : `~astropy.units.Quantity`
            Highest energy bin
        bins : int
            Number of bins

        Returns
        -------
        Energy
        """

        if not isinstance(emin, Quantity) or not isinstance(emax, Quantity):
            raise ValueError("Energies must be Quantities")

        x_min, x_max = np.log10([emin.value, emax.value])
        energy = np.logspace(x_min, x_max, nbins + 1)
        energy = Quantity(energy, emax.unit)

        return EnergyBounds(energy)

    @staticmethod
    def from_fits(hdulist):
        """Read EBOUNDS fits extension.

        Parameters
        ---------
        hdu_list : `~astropy.io.fits.HDUList` 
            HDU list with ``EBOUNDS`` extensions.

        Returns
        ------
        Energy
        """

        energy = Quantity(hdulist['EBOUNDS'].data['Energy'], 'MeV')
        return EnergyBounds(energy)

    def to_fits(self, **kwargs):
        """Write EBOUNDS fits extension

        Returns
        -------
        hdu_list : `~astropy.io.fits.BinTableHDU`
            EBOUNDS fits extension
        """

        col1 = fits.Column(name='Energy', format='D', array=self.value)
        cols = fits.ColDefs([col1])
        return fits.BinTableHDU.from_columns(cols)
