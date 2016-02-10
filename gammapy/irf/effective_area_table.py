# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle

from ..utils.energy import Energy
from ..extern.validator import validate_physical_type
from ..utils.array import array_stats_str
from ..utils.fits import table_to_fits_table, get_hdu_with_valid_name

__all__ = [
    'abramowski_effective_area',
    'EffectiveAreaTable',
    'EffectiveAreaTable2D',
]

log = logging.getLogger(__name__)


def abramowski_effective_area(energy, instrument='HESS'):
    """Simple IACT effective area parametrizations from Abramowski et al. (2010).

    TODO: give formula

    Parametrizations of the effective areas of Cherenkov telescopes
    taken from Appendix B of http://adsabs.harvard.edu/abs/2010MNRAS.402.1342A .

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Energy
    instrument : {'HESS', 'HESS2', 'CTA'}
        Instrument name

    Returns
    -------
    effective_area : `~astropy.units.Quantity`
        Effective area in cm^2
    """
    # Put the parameters g in a dictionary.
    # Units: g1 (cm^2), g2 (), g3 (MeV)
    # Note that whereas in the paper the parameter index is 1-based,
    # here it is 0-based
    pars = {'HESS': [6.85e9, 0.0891, 5e5],
            'HESS2': [2.05e9, 0.0891, 1e5],
            'CTA': [1.71e11, 0.0891, 1e5]}

    if not isinstance(energy, Quantity):
        raise ValueError("energy must be a Quantity object.")

    energy = energy.to('MeV').value

    if instrument not in pars.keys():
        ss = 'Unknown instrument: {0}\n'.format(instrument)
        ss += 'Valid instruments: HESS, HESS2, CTA'
        raise ValueError(ss)

    g1 = pars[instrument][0]
    g2 = pars[instrument][1]
    g3 = -pars[instrument][2]
    value = g1 * energy ** (-g2) * np.exp(g3 / energy)
    return Quantity(value, 'cm^2')


class EffectiveAreaTable(object):
    """
    Effective area table class.

    Not interpolation is used.

    Parameters
    ----------
    energy_lo : `~astropy.units.Quantity`
        Lower energy boundary of the energy bin.
    energy_hi : `~astropy.units.Quantity`
        Upper energy boundary of the energy bin.
    effective_area : `~astropy.units.Quantity`
        Effective area at the given energy bins.
    energy_thresh_lo : `~astropy.units.Quantity`
        Lower save energy threshold of the psf.
    energy_thresh_hi : `~astropy.units.Quantity`
        Upper save energy threshold of the psf.

    Examples
    --------
    Plot effective area vs. energy:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EffectiveAreaTable
        from gammapy.datasets import gammapy_extra
        filename = gammapy_extra.filename('test_datasets/unbundled/irfs/arf.fits')
        arf = EffectiveAreaTable.read(filename)
        arf.plot_area_vs_energy(show_safe_energy=False)
        plt.show()
    """

    def __init__(self, energy_lo, energy_hi, effective_area,
                 energy_thresh_lo=Quantity(0.1, 'TeV'),
                 energy_thresh_hi=Quantity(100, 'TeV')):

        # Validate input
        validate_physical_type('energy_lo', energy_lo, 'energy')
        validate_physical_type('energy_hi', energy_hi, 'energy')
        validate_physical_type('effective_area', effective_area, 'area')
        validate_physical_type('energy_thresh_lo', energy_thresh_lo, 'energy')
        validate_physical_type('energy_thresh_hi', energy_thresh_hi, 'energy')

        # Set attributes
        self.energy_hi = energy_hi.to('TeV')
        self.energy_lo = energy_lo.to('TeV')
        self.effective_area = effective_area.to('m^2')
        self.energy_thresh_lo = energy_thresh_lo.to('TeV')
        self.energy_thresh_hi = energy_thresh_hi.to('TeV')

    def to_table(self):
        """Convert to `~astropy.table.Table`.
        """
        table = Table()

        table['ENERG_LO'] = self.energy_lo
        table['ENERG_HI'] = self.energy_hi
        table['SPECRESP'] = self.effective_area

        return table

    def to_fits(self, header=None, energy_unit='TeV', effarea_unit='m2', **kwargs):
        """
        Convert ARF to FITS HDU list format.

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            Header to be written in the fits file.
        energy_unit : str
            Unit in which the energy is written in the fits file
        effarea__u : str
            Unit in which the effective area is written in the fits file

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            ARF in HDU list format.

        Notes
        -----
        For more info on the ARF FITS file format see:
        http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html

        Recommended units for ARF tables in X-ray astronomy are keV and cm^2,
        but TeV and m^2 are chosen here by default, as they are the more
        natural units for IACTs
        """

        self.energy_lo = self.energy_lo.to(energy_unit)
        self.energy_hi = self.energy_hi.to(energy_unit)
        self.effective_area = self.effective_area.to(effarea_unit)

        hdu = table_to_fits_table(self.to_table())

        if header is None:
            header = hdu.header

            # Write FITS extension header
            header['EXTNAME'] = 'SPECRESP', 'Name of this binary table extension'
            header['TELESCOP'] = 'DUMMY', 'Mission/satellite name'
            header['INSTRUME'] = 'DUMMY', 'Instrument/detector'
            header['FILTER'] = 'NONE', 'Filter information'
            header['HDUCLASS'] = 'OGIP', 'Organisation devising file format'
            header['HDUCLAS1'] = 'RESPONSE', 'File relates to response of instrument'
            header['HDUCLAS2'] = 'SPECRESP', 'Effective area data is stored'
            header['HDUVERS '] = '1.1.0', 'Version of file format'

            header['PHAFILE'] = ('', 'PHA file for which ARF was produced')

            # Obsolete ARF headers, included for the benefit of old software
            header['ARFVERSN'] = '1992a', 'Obsolete'
            header['HDUVERS1'] = '1.0.0', 'Obsolete'
            header['HDUVERS2'] = '1.1.0', 'Obsolete'

        for key, value in kwargs.items():
            header[key] = value

        header['LO_THRES'] = self.energy_thresh_lo.value
        header['HI_THRES'] = self.energy_thresh_hi.value

        prim_hdu = fits.PrimaryHDU()
        hdu.header = header
        return fits.HDUList([prim_hdu, hdu])

    def write(self, filename, energy_unit='TeV', effarea_unit='m2',
              *args, **kwargs):
        """Write ARF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits(energy_unit=energy_unit, effarea_unit=effarea_unit).writeto(
            filename, *args, **kwargs)

    @classmethod
    def read(cls, filename):
        """Create `EffectiveAreaTable` from ARF-format FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        hdu_list = fits.open(filename)
        return cls.from_fits(hdu_list)

    @classmethod
    def from_fits(cls, hdu_list):
        """Create `EffectiveAreaTable` from HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``SPECRESP`` extensions.

        Notes
        -----
        For more info on the ARF FITS file format see:
        http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html

        Recommended units for ARF tables are keV and cm^2,
        but TeV and m^2 are chosen here as the more natural units for IACTs.
        """
        energy_lo = Quantity(hdu_list['SPECRESP'].data['ENERG_LO'], 'TeV')
        energy_hi = Quantity(hdu_list['SPECRESP'].data['ENERG_HI'], 'TeV')
        effective_area = Quantity(hdu_list['SPECRESP'].data['SPECRESP'], 'm^2')
        try:
            energy_thresh_lo = Quantity(
                hdu_list['SPECRESP'].header['LO_THRES'], 'TeV')
            energy_thresh_hi = Quantity(
                hdu_list['SPECRESP'].header['HI_THRES'], 'TeV')
            return EffectiveAreaTable(energy_lo, energy_hi, effective_area,
                                      energy_thresh_lo, energy_thresh_hi)
        except KeyError:
            log.warning('No safe energy thresholds found. Setting to default')
            return cls(energy_lo, energy_hi, effective_area)

    def effective_area_at_energy(self, energy):
        """
        Get effective area for given energy.

        Not interpolation is used.

        Parameters
        ----------
        energy : `~astropy.units.Quantity`
            Energy where to return effective area.

        Returns
        -------
        effective_area : `~astropy.units.Quantity`
            Effective area at given energy.
        """
        if not isinstance(energy, Quantity):
            raise ValueError("energy must be a Quantity object.")

        # TODO: Use some kind of interpolation here
        i = np.argmin(np.abs(self.energy_hi - energy))
        return self.effective_area[i]

    def info(self):
        """Print some basic info.
        """
        ss = "\nSummary ARF info\n"
        ss += "----------------\n"
        # Summarise data members
        ss += array_stats_str(self.energy_lo, 'Energy lo')
        ss += array_stats_str(self.energy_hi, 'Energy hi')
        ss += array_stats_str(self.effective_area.to('m^2'), 'Effective area')
        ss += 'Safe energy threshold lo: {0:6.3f}\n'.format(self.energy_thresh_lo)
        ss += 'Safe energy threshold hi: {0:6.3f}\n'.format(self.energy_thresh_hi)

        return ss

    def plot_area_vs_energy(self, ax=None, show_safe_energy=True):
        """
        Plot effective area vs. energy.
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        energy_hi = self.energy_hi.value
        effective_area = self.effective_area.value
        ax.plot(energy_hi, effective_area)
        if show_safe_energy:
            ax.vlines(self.energy_thresh_hi.value, 1E3, 1E7, 'k', linestyles='--')
            text = 'Safe energy threshold: {0:3.2f}'.format(self.energy_thresh_hi)
            ax.text(self.energy_thresh_hi.value - 1, 3E6, text, ha='right')
            ax.vlines(self.energy_thresh_lo.value, 1E3, 1E7, 'k', linestyles='--')
            text = 'Safe energy threshold: {0:3.2f}'.format(self.energy_thresh_lo)
            ax.text(self.energy_thresh_lo.value + 0.1, 3E3, text)
        ax.set_xlim(0.1, 100)
        ax.set_ylim(1E3, 1E7)
        ax.set_xlabel('Energy (TeV)')
        ax.set_ylabel('Effective Area (m2)')
        ax.loglog()


class EffectiveAreaTable2D(object):
    """Offset-dependent radially-symmetric table effective area.

    Two interpolation methods area available:

    * `~scipy.interpolate.LinearNDInterpolator` (default)
    * `~scipy.interpolate.RectBivariateSpline`

    Equivalent to GammaLib/ctools``GCTAAeff2D FITS`` format

    Parameters
    ----------
    energy_lo, energy_hi : `~astropy.units.Quantity`
        Lower / upper energy bin edges vector
    offset_lo, offset_hi : `~astropy.coordinates.Angle`
        Lower / upper offset bin edges vector
    eff_area : `~astropy.units.Quantity`
        Effective area vector (true energy)
    method : str
        Interpolation method


    Examples
    --------
    Get effective area vs. energy for a given offset and energy binning:

    .. code-block:: python

        import numpy as np
        from astropy.coordinates import Angle
        from astropy.units import Quantity
        from gammapy.irf import EffectiveAreaTable2D
        from gammapy.datasets import gammapy_extra
        filename = gammapy_extra.filename('test_datasets/unbundled/irfs/aeff2D.fits')
        aeff2D = EffectiveAreaTable2D.read(filename)
        offset = Angle(0.6, 'deg')
        energy = Quantity(np.logspace(0, 1, 60), 'TeV')
        eff_area = aeff2D.evaluate(offset, energy)

    Create ARF fits file for a given offset and energy binning:

    .. code-block:: python

        import numpy as np
        from astropy.coordinates import Angle
        from astropy.units import Quantity
        from gammapy.irf import EffectiveAreaTable2D
        from gammapy.utils.energy import EnergyBounds
        from gammapy.datasets import gammapy_extra
        filename = gammapy_extra.filename('test_datasets/unbundled/irfs/aeff2D.fits')
        aeff2D = EffectiveAreaTable2D.read(filename)
        offset = Angle(0.43, 'deg')
        nbins = 50
        energy = EnergyBounds.equal_log_spacing(1, 10, nbins, 'TeV')
        energy_lo = energy[:-1]
        energy_hi = energy[1:]
        arf_table = aeff2D.to_effective_area_table(offset, energy_lo, energy_hi)
        arf_table.write('arf.fits')

    Plot energy dependence

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EffectiveAreaTable2D
        from gammapy.datasets import gammapy_extra
        filename = gammapy_extra.filename('test_datasets/unbundled/irfs/aeff2D.fits')
        aeff2D = EffectiveAreaTable2D.read(filename)
        aeff2D.plot_energy_dependence()
        plt.loglog()
        plt.xlim(0.8, 100)
        plt.ylim(2E4, 2E6)

    """

    def __init__(self, energy_lo, energy_hi, offset_lo,
                 offset_hi, eff_area, method='linear', thres_lo=None,
                 thres_hi=None):

        if not isinstance(energy_lo, Quantity) or not isinstance(energy_hi, Quantity):
            raise ValueError("Energies must be Quantity objects.")
        if not isinstance(offset_lo, Angle) or not isinstance(offset_hi, Angle):
            raise ValueError("Offsets must be Angle objects.")
        if not isinstance(eff_area, Quantity):
            raise ValueError("Effective areas must be Quantity objects.")

        self.energy_lo = energy_lo.to('TeV')
        self.energy_hi = energy_hi.to('TeV')
        self.offset_lo = offset_lo.to('deg')
        self.offset_hi = offset_hi.to('deg')
        self.eff_area = eff_area.to('m^2')
        self.offset = (offset_hi + offset_lo) / 2
        self.energy = np.sqrt(energy_lo * energy_hi)

        self._thres_lo = thres_lo
        self._thres_hi = thres_hi

        self._prepare_linear_interpolator()
        self._prepare_spline_interpolator()

        # set to linear interpolation by default
        self.interpolation_method = method

    @classmethod
    def from_fits(cls, hdu_list, column='true'):
        """Create `EffectiveAreaTable2D` from ``GCTAAeff2D`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``EFFECTIVE AREA`` extension.
        column : str {'true', 'reco'}
            Effective area column to be read
        """
        # Locate an HDU with the right name or raise and error
        hdu = get_hdu_with_valid_name(hdu_list, valid_extnames=['AEFF_2D', 'EFFECTIVE AREA'])

        data = hdu.data
        header = hdu.header
        thres_lo = Energy(header['LO_THRES'], 'TeV')
        thres_hi = Energy(header['HI_THRES'], 'TeV')
        e_lo = Quantity(data['ENERG_LO'].squeeze(), 'TeV')
        e_hi = Quantity(data['ENERG_HI'].squeeze(), 'TeV')
        o_lo = Angle(data['THETA_LO'].squeeze(), 'deg')
        o_hi = Angle(data['THETA_HI'].squeeze(), 'deg')
        if column == 'reco':
            ef = Quantity(data['EFFAREA'].squeeze(), 'm^2')
        else:
            ef = Quantity(data['EFFAREA_RECO'].squeeze(), 'm^2')

        return cls(e_lo, e_hi, o_lo, o_hi, ef, thres_lo=thres_lo, thres_hi=thres_hi)

    @classmethod
    def read(cls, filename, column='true'):
        """Create `EffectiveAreaTable2D` from ``GCTAAeff2D`` format FITS file.

        Parameters
        ----------
        filename : str
            File name
        column : str {'true', 'reco'}
            Effective area column to be read
        """
        hdu_list = fits.open(filename)
        return EffectiveAreaTable2D.from_fits(hdu_list, column=column)

    @property
    def low_threshold(self):
        """
        Low energy threshold
        """
        return self._thres_lo

    @property
    def high_threshold(self):
        """
        Low energy threshold
        """
        return self._thres_hi

    def to_effective_area_table(self, offset, energy_lo=None, energy_hi=None):
        """Evaluate at a given offset and return effective area table.

        The energy thresholds in the effective area table object are not set.
        If the effective area table is intended to be used for spectral analysis,
        the final true energy binning should be given here, since the
        effective area table class does no interpolation.

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            offset
        energy_lo, energy_hi : `~astropy.units.Quantity`
            Energy lower and upper bounds array

        Returns
        -------
        eff_area_table : `EffectiveAreaTable`
             Effective area table
        """
        offset = Angle(offset)

        if energy_lo is None and energy_hi is None:
            energy_lo = self.energy_lo
            energy_hi = self.energy_hi
        elif energy_lo is None or energy_hi is None:
            raise ValueError("Only 1 energy vector given, need 2")
        if not isinstance(energy_lo, Quantity) or not isinstance(energy_hi, Quantity):
            raise ValueError("Energy must be a Quantity object.")
        if len(energy_lo) != len(energy_hi):
            raise ValueError("Energy Vectors must have same length")

        energy = np.sqrt(energy_lo * energy_hi)
        area = self.evaluate(offset, energy)

        return EffectiveAreaTable(energy_lo, energy_hi, area,
                                  energy_thresh_lo=self.low_threshold,
                                  energy_thresh_hi=self.high_threshold)

    def evaluate(self, offset=None, energy=None):
        """Evaluate effective area for a given energy and offset.

        If a parameter is not given, the nodes from the FITS table are used.
        Both offset and energy can be arbitrary ndarrays

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            offset
        energy : `~astropy.units.Quantity`
            energy

        Returns
        -------
        eff_area : `~astropy.units.Quantity`
            Effective Area

        """

        if offset is None:
            offset = self.offset
        if energy is None:
            energy = self.energy
        if not isinstance(energy, Quantity):
            raise ValueError("Energy must be a Quantity object.")
        if not isinstance(offset, Angle):
            raise ValueError("Offset must be an Angle object.")

        offset = offset.to('deg')
        energy = energy.to('TeV')

        method = self.interpolation_method
        if method == 'linear':
            val = self._eval_linear(offset.value, np.log10(energy.value))
        elif method == 'spline':
            val = self._eval_spline(offset.value, np.log10(energy.value))
        else:
            raise ValueError('Invalid interpolation method: {}'.format(method))
        return Quantity(val, self.eff_area.unit)

    def _eval_linear(self, offset=None, energy=None):
        """Evaluate linear interpolator

        Parameters
        ----------
        offset : float or ndarray of floats
            offset in deg
        energy : float or ndarray of floats
            energy in TeV

        Returns
        -------
        eff_area : float or ndarray of floats
            Effective area
        """
        off = np.atleast_1d(offset)
        ener = np.atleast_1d(energy)
        shape_off = np.shape(off)
        shape_ener = np.shape(ener)
        off = off.flatten()
        ener = ener.flatten()
        points = [(x, y) for x in off for y in ener]
        val = self._linear(points)
        val = np.reshape(val, np.append(shape_off, shape_ener)).squeeze()

        return val

    def _eval_spline(self, offset=None, energy=None):
        """Evaluate spline interpolator

        Not implemented
        """
        raise NotImplementedError

    def plot_energy_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot effective area versus energy for a given offset.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset = Angle(np.linspace(0.5, 2, 4), 'deg')

        if energy is None:
            energy = self.energy

        for off in offset:
            area = self.evaluate(off, energy)
            label = 'offset = {:.1f}'.format(off)
            ax.plot(energy, area.value, label=label, **kwargs)

        ax.loglog()
        ax.set_ylim(1e2, 1e7)
        ax.set_xlabel('Energy ({0})'.format(self.energy.unit))
        ax.set_ylabel('Effective Area ({0})'.format(self.eff_area.unit))
        ax.legend(loc='lower right')

        return ax

    def plot_offset_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot effective area versus offset for a given energy
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if energy is None:
            energy = Quantity(np.logspace(-1, 2, 8), 'TeV')

        if offset is None:
            off_lo = self.offset[0].to('deg').value
            off_hi = self.offset[-1].to('deg').value
            offset = Angle(np.linspace(off_lo, off_hi, 100), 'deg')

        for ee in energy:
            area = self.evaluate(offset, ee)
            area /= np.nanmax(area)
            if np.isnan(area).all():
                continue
            label = 'energy = {:.1f}'.format(ee)
            ax.plot(offset, area, label=label, **kwargs)

        ax.set_ylim(0, 1.1)
        ax.set_xlabel('Offset ({0})'.format(self.offset.unit))
        ax.set_ylabel('Relative Effective Area')
        ax.legend(loc='best')

        return ax

    def plot_image(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot effective area image (x=offset, y=energy).
        """
        import matplotlib.pyplot as plt

        kwargs.setdefault('cmap', 'afmhot')
        kwargs.setdefault('origin', 'bottom')
        kwargs.setdefault('interpolation', 'nearest')

        ax = plt.gca() if ax is None else ax

        if offset is None:
            vals = self.offset.value
            offset = np.linspace(vals.min(), vals.max(), 100)
            offset = Angle(offset, self.offset.unit)

        if energy is None:
            vals = self.energy.value
            energy = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 100)
            energy = Quantity(energy, self.energy.unit)

        aeff = self.evaluate(offset, energy).T
        extent = [
            offset.value.min(), offset.value.max(),
            energy.value.min(), energy.value.max(),
        ]
        ax.imshow(aeff.value, extent=extent, **kwargs)
        # ax.set_xlim(offset.value.min(), offset.value.max())
        # ax.set_ylim(energy.value.min(), energy.value.max())

        ax.semilogy()
        ax.set_xlabel('Offset ({0})'.format(offset.unit))
        ax.set_ylabel('Energy ({0})'.format(energy.unit))
        ax.set_title('Effective Area ({0})'.format(aeff.unit))
        ax.legend()

        return ax

    def peek(self, figsize=(15, 5)):
        """Quick-look summary plots."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.plot_image(ax=axes[0])
        self.plot_energy_dependence(ax=axes[1])
        self.plot_offset_dependence(ax=axes[2])
        plt.tight_layout()
        plt.show()

    def info(self):
        """Print some basic info.
        """
        ss = "\nSummary EffectiveArea2D info\n"
        ss += "----------------\n"
        # Summarise data members
        ss += array_stats_str(self.energy, 'energy')
        ss += array_stats_str(self.offset, 'offset')
        ss += array_stats_str(self.eff_area, 'dispersion')

        return ss

    def _prepare_linear_interpolator(self):
        """Setup `~scipy.interpolate.RegularGridInterpolator`
        """

        from scipy.interpolate import RegularGridInterpolator

        x = self.offset.value
        y = np.log10(self.energy.value)
        vals = self.eff_area.value

        self._linear = RegularGridInterpolator((x, y), vals, bounds_error=True)

    def _prepare_spline_interpolator(self):
        """Only works for radial symmetric input files (N=2)
        """

        # TODO Replace by scipy.ndimage.interpolation.map_coordinates

        from scipy.interpolate import RectBivariateSpline

        x = self.offset.value
        y = np.log10(self.energy.value)

        self._spline = RectBivariateSpline(x, y, self.eff_area.value)
