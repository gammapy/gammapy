# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy import log
from astropy.io import fits
from astropy.units import Quantity
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from ..extern.validator import validate_physical_type
from ..utils.array import array_stats_str

__all__ = ['abramowski_effective_area', 'EffectiveAreaTable',
           'EffectiveAreaTable2D']


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
        from gammapy.datasets import load_arf_fits_table
        arf = EffectiveAreaTable.from_fits(load_arf_fits_table())
        arf.plot_area_vs_energy(show_save_energy=False)
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

    def to_fits(self, header=None, energy_unit='TeV', effarea_unit='m2', **kwargs):
        """
        Convert ARF to FITS HDU list format.

        Parameters
        ----------
        header : `~astropy.io.fits.header.Header`
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

        hdu = fits.new_table(
            [fits.Column(name='ENERG_LO',
                         format='1E',
                         array=self.energy_lo.value,
                         unit=str(self.energy_lo.unit)),
             fits.Column(name='ENERG_HI',
                         format='1E',
                         array=self.energy_hi.value,
                         unit=str(self.energy_hi.unit)),
             fits.Column(name='SPECRESP',
                         format='1E',
                         array=self.effective_area.value,
                         unit=str(self.effective_area.unit))
             ]
        )

        if header is None:
            from ..datasets import load_arf_fits_table
            header = load_arf_fits_table()[1].header

        if header == 'pyfact':
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
        hdu.add_checksum()
        hdu.add_datasum()
        return fits.HDUList([prim_hdu, hdu])

    def write(self, filename, *args, **kwargs):
        """Write ARF to FITS file.

        Calls `~astropy.io.fits.HDUList.writeto`, forwarding all arguments.
        """
        self.to_fits().writeto(filename, *args, **kwargs)

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
            log.warn('No safe energy thresholds found. Setting to default')
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

    def info(self, energies=Quantity([1, 10], 'TeV')):
        """
        Print some ARF info.

        Parameters
        ----------
        energies : `~astropy.units.Quantity`
            Energies for which to print effective areas.
        """
        ss = "\nSummary ARF info\n"
        ss += "----------------\n"
        # Summarise data members
        ss += array_stats_str(self.energy_lo, 'Energy lo')
        ss += array_stats_str(self.energy_hi, 'Energy hi')
        ss += array_stats_str(self.effective_area.to('m^2'), 'Effective area')
        ss += 'Safe energy threshold lo: {0:6.3f}\n'.format(self.energy_thresh_lo)
        ss += 'Safe energy threshold hi: {0:6.3f}\n'.format(self.energy_thresh_hi)

        for energy in energies:
            eff_area = self.effective_area_at_energy(energy)
            ss += ("Effective area at E = {0:4.1f}: {1:10.0f}\n".format(energy, eff_area))
        return ss

    def plot_area_vs_energy(self, filename=None, show_save_energy=True):
        """
        Plot effective area vs. energy.
        """
        import matplotlib.pyplot as plt

        energy_hi = self.energy_hi.value
        effective_area = self.effective_area.value
        plt.plot(energy_hi, effective_area)
        if show_save_energy:
            plt.vlines(self.energy_thresh_hi.value, 1E3, 1E7, 'k', linestyles='--')
            plt.text(self.energy_thresh_hi.value - 1, 3E6,
                     'Safe energy threshold: {0:3.2f}'.format(
                         self.energy_thresh_hi),
                     ha='right')
            plt.vlines(self.energy_thresh_lo.value, 1E3, 1E7, 'k', linestyles='--')
            plt.text(self.energy_thresh_lo.value + 0.1, 3E3,
                     'Safe energy threshold: {0:3.2f}'.format(self.energy_thresh_lo))
        plt.xlim(0.1, 100)
        plt.ylim(1E3, 1E7)
        plt.loglog()
        plt.xlabel('Energy [TeV]')
        plt.ylabel('Effective Area [m^2]')
        if filename is not None:
            plt.savefig(filename)
            log.info('Wrote {0}'.format(filename))


class EffectiveAreaTable2D(object):
    """Offset-dependent radially-symmetric table effective area.

    Two interpolation methods area available:

    * `~scipy.interpolate.LinearNDInterpolator` (default)
    * `~scipy.interpolate.RectBivariateSpline`

    Equivalent to GammaLib/ctools``GCTAAeff2D FITS`` format

    Parameters
    ----------
    energ_lo : `~astropy.units.Quantity`
        Lower energy bin edges vector
    energ_hi : `~astropy.units.Quantity`
        Upper energy bin edges vector
    offset_lo : `~astropy.coordinates.Angle`
        Lower offset bin edges vector
    offset_hi : `~astropy.coordinates.Angle`
        Upper offset bin edges  vector
    eff_area : `~astropy.units.Quantity`
        Effective area vector (true energy)
    eff_area_reco : `~astropy.units.Quantity`
        Effective area vector (reconstructed energy)
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
        from gammapy.datasets import load_aeff2D_fits_table
        aeff2D = EffectiveAreaTable2D.from_fits(load_aeff2D_fits_table())
        offset = Angle(0.6, 'degree')
        energy = Quantity(np.logspace(0, 1, 60), 'TeV')
        eff_area = aeff2D.evaluate(offset, energy)

    Create ARF fits file for a given offest and energy binning:

    .. code-block:: python
    
        import numpy as np
        from astropy.coordinates import Angle
        from astropy.units import Quantity
        from gammapy.irf import EffectiveAreaTable2D
        from gammapy.spectrum import energy_bounds_equal_log_spacing
        from gammapy.datasets import load_aeff2D_fits_table
        aeff2D = EffectiveAreaTable2D.from_fits(load_aeff2D_fits_table())
        offset = Angle(0.43, 'degree')
        nbins = 50
        energy = energy_bounds_equal_log_spacing(Quantity((1,10), 'TeV'), nbins)
        energ_lo = energy[:-1]
        energ_hi = energy[1:]
        arf_table = aeff2D.to_effective_area_table(offset, energ_lo, energ_hi) 
        arf_table.write('arf.fits')

    Plot energy dependence

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EffectiveAreaTable2D
        from gammapy.datasets import load_aeff2D_fits_table
        aeff2D = EffectiveAreaTable2D.from_fits(load_aeff2D_fits_table())
        aeff2D.plot_energy_dependence()

    """

    def __init__(self, energ_lo, energ_hi, offset_lo,
                 offset_hi, eff_area, eff_area_reco, method='linear'):
        if not isinstance(energ_lo, Quantity) or not isinstance(energ_hi, Quantity):
            raise ValueError("Energies must be Quantity objects.")
        if not isinstance(offset_lo, Angle) or not isinstance(offset_hi, Angle):
            raise ValueError("Offsets must be Angle objects.")
        if not isinstance(eff_area, Quantity) or not isinstance(eff_area_reco, Quantity):
            raise ValueError("Effective areas must be Quantity objects.")

        self.energ_lo = energ_lo.to('TeV')
        self.energ_hi = energ_hi.to('TeV')
        self.offset_lo = offset_lo.to('degree')
        self.offset_hi = offset_hi.to('degree')
        self.eff_area = eff_area.to('m^2')
        self.eff_area_reco = eff_area_reco.to('m^2')

        # actually offset_hi = offset_lo
        self.offset = (offset_hi + offset_lo) / 2
        self.energy = np.sqrt(energ_lo * energ_hi)

        self._prepare_linear_interpolator()
        self._prepare_spline_interpolator()

        # set to linear interpolation by default
        self.interpolation_method = method

    @classmethod
    def from_fits(cls, hdu_list):
        """Create `EffectiveAreaTable2D` from ``GCTAAeff2D`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``EFFECTIVE AREA`` extension.
        """

        data = hdu_list['EFFECTIVE AREA'].data
        e_lo = Quantity(data['ENERG_LO'].squeeze(), 'TeV')
        e_hi = Quantity(data['ENERG_HI'].squeeze(), 'TeV')
        o_lo = Angle(data['THETA_LO'].squeeze(), 'degree')
        o_hi = Angle(data['THETA_HI'].squeeze(), 'degree')
        ef = Quantity(data['EFFAREA'].squeeze(), 'm^2')
        efrec = Quantity(data['EFFAREA_RECO'].squeeze(), 'm^2')

        return cls(e_lo, e_hi, o_lo, o_hi, ef, efrec)

    @classmethod
    def read(cls, filename):
        """Create `EffectiveAreaTable2D` from ``GCTAAeff2D`` format FITS file.

        Parameters
        ----------
        filename : str
            File name
        """
        hdu_list = fits.open(filename)
        return EffectiveAreaTable2D.from_fits(hdu_list)

    def to_effective_area_table(self, offset, energy_lo=None, energy_hi=None):
        """Evaluate at a given offset and return effective area table.

        The energy thresholds in the effective area table object are not set.
        If the effective area table is intended to be used for spectral analysis,
        the final energy binning should be given here, since the
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

        if energy_lo is None and energy_hi is None:
            energy_lo = self.energy_lo
            energy_hi = self.energy_hi
        elif energy_lo is None or energy_hi is None:
            raise ValueError("Only 1 energy vector given, need 2")
        if not isinstance(energy_lo, Quantity) or not isinstance(energy_hi, Quantity):
            raise ValueError("Energy must be a Quantity object.")
        if len(energy_lo) != len(energy_hi):
            raise ValueError("Energy Vectors must have same length")

        if not isinstance(offset, Angle):
            raise ValueError("Offset must be an angle object.")

        energy = np.sqrt(energy_lo * energy_hi)
        area = self.evaluate(offset, energy)

        return EffectiveAreaTable(energy_lo, energy_hi, area)

    def evaluate(self, offset=None, energy=None):
        """Evaluate effective area for a given energy and offset.

        If a parameter is not given, the nodes from the FITS table are used.
        2D input arrays are not supported yet.

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

        offset = offset.to('degree')
        energy = energy.to('TeV')

        # support energy=1Darray & offset=1Darray
        if offset.shape != () and energy.shape != ():
            val = np.zeros([len(offset), len(energy)])
            for i in range(len(offset)):
                val[i] = self._eval(offset[i], energy)
        # default
        else:
            val = self._eval(offset=offset, energy=energy)

        return Quantity(val, self.eff_area.unit)

    def _eval(self, offset=None, energy=None):
        method = self.interpolation_method
        if(method == 'linear'):
            val = self._linear(offset.value, np.log10(energy.value))
        elif (method == 'spline'):
            val = self._spline(offset.value, np.log10(energy.value)).squeeze()
        else:
            raise ValueError('Invalid interpolation method: {}'.format(method))
        return Quantity(val, self.eff_area.unit)

    def plot_energy_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot effective area versus energy for a given offset.
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if offset is None:
            val = self.offset
            offset = np.linspace(np.min(val), np.max(val), 5)

        if energy is None:
            energy = self.energy

        for off in offset:
            area = self.evaluate(off, energy)
            label = 'offset = {:.1f}'.format(off)
            plt.plot(energy, area.value, label=label, **kwargs)

        plt.xlabel('Energy ({0})'.format(self.energy.unit))
        plt.ylabel('Effective Area ({0})'.format(self.eff_area.unit))
        plt.legend()

        return ax

    def plot_offset_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot effective area versus offset for a given energy
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if energy is None:
            val = self.energy
            energy = np.linspace(np.min(val), np.max(val), 4)

        if offset is None:
            offset = self.offset

        for ener in energy:
            area = self.evaluate(offset, ener)
            label = 'energy = {:.1f}'.format(ener)
            plt.plot(offset, area.value, label=label, **kwargs)

        plt.xlabel('Offset ({0})'.format(self.offset.unit))
        plt.ylabel('Effective Area ({0})'.format(self.eff_area.unit))
        plt.legend()

        return ax

    def _prepare_linear_interpolator(self):
        """Could be generalized for non-radial symmetric input files (N>2)
        """

        from scipy.interpolate import LinearNDInterpolator

        x = self.offset.value
        y = np.log10(self.energy.value)
        coord = [(xx, yy) for xx in x for yy in y]

        vals = self.eff_area.value.flatten()
        self._linear = LinearNDInterpolator(coord, vals)

    def _prepare_spline_interpolator(self):
        """Only works for radial symmetric input files (N=2)
        """

        from scipy.interpolate import RectBivariateSpline

        x = self.offset.value
        y = np.log10(self.energy.value)

        self._spline = RectBivariateSpline(x, y, self.eff_area.value)
