# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.units import Quantity
from astropy.coordinates import Angle

from ..utils.energy import Energy, EnergyBounds
from ..extern.validator import validate_physical_type
from ..utils.array import array_stats_str
from ..utils.fits import table_to_fits_table
from ..utils.scripts import make_path

__all__ = [
    'abramowski_effective_area',
    'EffectiveAreaTable',
    'EffectiveAreaTable2D',
]

log = logging.getLogger(__name__)


class EffectiveAreaTable(object):
    """
    Effective area table class.

    Not interpolation is used.

    Parameters
    ----------
    ebounds : `~gammapy.utils.energy.EnergyBounds`
        Energy axis
    effective_area : `~astropy.units.Quantity`
        Effective area at the given energy bins.
    energy_thresh_lo : `~astropy.units.Quantity`
        Lower save energy threshold of the psf.
    energy_thresh_hi : `~astropy.units.Quantity`
        Upper save energy threshold of the psf.

    Examples
    --------
    Plot effective area versus energy:

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from gammapy.irf import EffectiveAreaTable
        from gammapy.datasets import gammapy_extra
        filename = gammapy_extra.filename('datasets/hess-crab4_pha/ogip_data/arf_run23523.fits')
        arf = EffectiveAreaTable.read(filename)
        arf.plot(show_safe_energy=True)
        plt.show()
    """

    def __init__(self, ebounds, effective_area,
                 energy_thresh_lo=None, energy_thresh_hi=None):

        if energy_thresh_lo is not None:
            self.energy_thresh_lo = Energy(energy_thresh_lo).to('TeV')
        else:
            self.energy_thresh_lo = Energy('10 GeV')
        if energy_thresh_hi is not None:
            self.energy_thresh_hi = Energy(energy_thresh_hi).to('TeV')
        else:
            self.energy_thresh_hi = Energy('100 TeV')

        # Validate input
        validate_physical_type('effective_area', effective_area, 'area')

        # Set attributes
        self.ebounds = EnergyBounds(ebounds)
        self.effective_area = effective_area

    def to_table(self):
        """Convert to `~astropy.table.Table`.
        """
        table = Table()

        table['ENERG_LO'] = self.ebounds.lower_bounds
        table['ENERG_HI'] = self.ebounds.upper_bounds
        table['SPECRESP'] = self.effective_area

        return table

    def to_fits(self, header=None, energy_unit='TeV', effarea_unit='m2', **kwargs):
        """
        Convert ARF to FITS HDU list format.

        For more info on the ARF FITS file format see :ref:`gadf:ogip-arf`

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
        """

        self.ebounds = self.ebounds.to(energy_unit)
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

        For more info on the ARF FITS file format see :ref:`gadf:ogip-arf`

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.HDUList`
            HDU list with ``SPECRESP`` extensions.
        """
        spec = hdu_list['SPECRESP']
        e_unit = spec.header['TUNIT1']
        a_unit = spec.header['TUNIT3']

        e_lo = Quantity(spec.data['ENERG_LO'], e_unit)
        e_hi = Quantity(spec.data['ENERG_HI'], e_unit)
        ebounds = EnergyBounds.from_lower_and_upper_bounds(e_lo, e_hi)
        effective_area = Quantity(hdu_list['SPECRESP'].data['SPECRESP'], a_unit)
        e_thresh_hi = None
        e_thresh_lo = None
        try:
            e_thresh_lo = Quantity(
                hdu_list['SPECRESP'].header['LO_THRES'], 'TeV')
            e_thresh_hi = Quantity(
                hdu_list['SPECRESP'].header['HI_THRES'], 'TeV')
        except KeyError:
            log.warning('No safe energy thresholds found. Setting to default')

        return cls(ebounds, effective_area, e_thresh_lo, e_thresh_hi)

    def evaluate(self, energy=None):
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
        energy = self.ebounds.log_centers if energy is None else Energy(energy)

        i = self.ebounds.find_energy_bin(energy)

        # TODO: Use some kind of interpolation here
        return self.effective_area[i]

    def info(self):
        """Print some basic info.
        """
        ss = "\nSummary ARF info\n"
        ss += "----------------\n"
        # Summarise data members
        ss += array_stats_str(self.ebounds.to('TeV'), 'Energy Bounds')
        ss += array_stats_str(self.effective_area.to('m2'), 'Effective area')
        ss += 'Safe energy threshold lo: {0:6.3f}\n'.format(self.energy_thresh_lo)
        ss += 'Safe energy threshold hi: {0:6.3f}\n'.format(self.energy_thresh_hi)

        return ss

    def plot(self, ax=None, show_safe_energy=True, energy=None, energy_unit='TeV',
             eff_area_unit='m2', **kwargs):
        """
        Plot effective area

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        show_safe_energy : bool
            Show safe energy range on the plot
        energy : `~astropy.units.Quantity`
            Energy where to plot effective area.

        Returns
        -------
        ax : `~matplolib.axes`
            Axis

        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        kwargs.setdefault('lw', 2)

        energy = self.ebounds.log_centers if energy is None else Energy(energy)
        eff_area = self.evaluate(energy)

        energy = energy.to(energy_unit).value
        eff_area = eff_area.to(eff_area_unit).value

        ax.plot(energy, eff_area, **kwargs)
        if show_safe_energy:
            ymin, ymax = ax.get_ylim()
            line_kwargs = dict(lw=2, color='black')
            ax.vlines(self.energy_thresh_lo.value, ymin, ymax, linestyle='dashed',
                      label='Low energy threshold {:.2f}'.format(self.energy_thresh_lo),
                      **line_kwargs)
            ax.vlines(self.energy_thresh_hi.value, ymin, ymax, linestyle='dotted',
                      label='High energy threshold {:.2f}'.format(self.energy_thresh_hi),
                      **line_kwargs)
            ax.legend(loc='upper left')

        ax.set_xscale('log')
        ax.set_xlabel('Energy [{}]'.format(energy_unit))
        ax.set_ylabel('Effective Area [{}]'.format(eff_area_unit))

        return ax


class EffectiveAreaTable2D(object):
    """Offset-dependent radially-symmetric table effective area.

    Two interpolation methods area available:

    * `~scipy.interpolate.RegularGridInterpolator` (default)
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
    interp_kwargs : dict or None
        Interpolation parameter dict passed to `scipy.interpolate.RegularGridInterpolator`.
        If you pass ``None``, the default ``interp_params=dict(bounds_error=False)`` is used.


    Examples
    --------
    Get effective area as a function of energy for a given offset and energy binning:

    .. code-block:: python

        from gammapy.irf import EffectiveAreaTable2D
        from gammapy.utils.energy import EnergyBounds
        from gammapy.datasets import gammapy_extra
        filename = gammapy_extra.filename('hess-crab4-hd-hap-prod2/run023400-023599/run023523/hess_aeff_2d_023523.fits.gz')
        aeff2D = EffectiveAreaTable2D.read(filename)
        energy = EnergyBounds.equal_log_spacing(1, 10, 60, 'TeV')
        eff_area = aeff2D.evaluate('0.6 deg', energy)

    Load EffectiveAreaTable2D from data store and create ARF fits file

    .. code-block:: python

        from gammapy.utils.testing import data_manager
        from astropy.coordinates import Angle
        from gammapy.utils.energy import EnergyBounds

        dm = data_manager()
        ds = dm['hess-crab4-hd-hap-prod2']
        aeff2D = ds.load(23523, filetype='aeff')
        offset = Angle(0.5, 'deg')
        nbins = 50
        energy = EnergyBounds.equal_log_spacing(1, 10, nbins, 'TeV')
        arf = aeff2D.to_effective_area_table(offset, energy)
        arf.write('arf.fits')

    Plot energy dependence for several offsets

    .. plot::
        :include-source:

        from gammapy.utils.testing import data_manager
        import matplotlib.pyplot as plt

        dm = data_manager()
        ds = dm['hess-crab4-hd-hap-prod2']
        aeff2D = ds.obs(obs_id=23523).aeff
        aeff2D.plot_energy_dependence()
    """

    def __init__(self, energy_lo, energy_hi, offset_lo,
                 offset_hi, eff_area, method='linear', interp_kwargs=None,
                 thres_lo=None, thres_hi=None):

        if not isinstance(energy_lo, Quantity) or not isinstance(energy_hi, Quantity):
            raise ValueError("Energies must be Quantity objects.")
        if not isinstance(offset_lo, Angle) or not isinstance(offset_hi, Angle):
            raise ValueError("Offsets must be Angle objects.")
        if not isinstance(eff_area, Quantity):
            raise ValueError("Effective areas must be Quantity objects.")

        self.ebounds = EnergyBounds.from_lower_and_upper_bounds(energy_lo, energy_hi)
        self.offset_lo = offset_lo.to('deg')
        self.offset_hi = offset_hi.to('deg')
        self.eff_area = eff_area.to('m^2')
        self.offset = (offset_hi + offset_lo) / 2

        self._thres_lo = thres_lo
        self._thres_hi = thres_hi

        if not interp_kwargs:
            interp_kwargs = dict(bounds_error=False, fill_value = None)

        self._prepare_linear_interpolator(interp_kwargs)
        self._prepare_spline_interpolator()

        # set to linear interpolation by default
        self.interpolation_method = method

    @classmethod
    def from_fits(cls, hdu, column='true'):
        """Create `EffectiveAreaTable2D` from ``GCTAAeff2D`` format HDU list.

        Parameters
        ----------
        hdu_list : `~astropy.io.fits.BinTableHDU`
            HDU
        column : str {'true', 'reco'}
            Effective area column to be read
        """
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
    def read(cls, filename, hdu='aeff_2d', column='true'):
        """Create `EffectiveAreaTable2D` from ``GCTAAeff2D`` format FITS file.

        Parameters
        ----------
        filename : str
            File name
        column : str {'true', 'reco'}
            Effective area column to be read
        """
        filename = make_path(filename)
        hdu_list = fits.open(str(filename))
        hdu = hdu_list[hdu]
        return EffectiveAreaTable2D.from_fits(hdu, column=column)

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

    def to_effective_area_table(self, offset, ebounds=None):
        """Evaluate at a given offset and return effective area table.

        If the effective area table is intended to be used for spectral analysis,
        the final true energy binning should be given here, since the
        effective area table class does no interpolation.

        Parameters
        ----------
        offset : `~astropy.coordinates.Angle`
            offset
        ebounds : `~gammapy.utils.energy.EnergyBounds`
            Energy axis

        Returns
        -------
        eff_area_table : `EffectiveAreaTable`
             Effective area table
        """

        offset = Angle(offset)
        ebounds = self.ebounds if ebounds is None else EnergyBounds(ebounds)
        area = self.evaluate(offset, ebounds.log_centers)
        return EffectiveAreaTable(ebounds, area,
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

        offset = self.offset if offset is None else Angle(offset)
        energy = self.ebounds.log_centers if energy is None else Energy(energy)

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

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`
            Offset
        energy : `~gammapy.utils.energy.Energy`
            Energy axis
        kwargs : dict
            Forwarded tp plt.plot()

        Returns
        -------
        ax : `~matplolib.axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if offset is None:
            offset = Angle(np.linspace(0.5, 2, 4), 'deg')

        if energy is None:
            energy = self.ebounds.log_centers

        for off in offset:
            area = self.evaluate(off, energy)
            label = 'offset = {:.1f}'.format(off)
            ax.plot(energy, area.value, label=label, **kwargs)

        ax.set_xscale('log')
        ax.set_xlabel('Energy [{0}]'.format(self.ebounds.unit))
        ax.set_ylabel('Effective Area [{0}]'.format(self.eff_area.unit))
        ax.set_xlim(min(energy.value), max(energy.value))
        ax.legend(loc='upper left')

        return ax

    def plot_offset_dependence(self, ax=None, offset=None, energy=None, **kwargs):
        """Plot effective area versus offset for a given energy

        Parameters
        ----------
        ax : `~matplolib.axes`, optional
            Axis
        offset : `~astropy.coordinates.Angle`
            Offset axis
        energy : `~gammapy.utils.energy.Energy`
            Energy 

        Returns
        -------
        ax : `~matplolib.axes`
            Axis
        """
        import matplotlib.pyplot as plt

        ax = plt.gca() if ax is None else ax

        if energy is None:
            emin = self.ebounds[1]
            emax = self.ebounds[-2]
            energy = Energy.equal_log_spacing(emin, emax, nbins=5)

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
            vals = self.ebounds.log_centers.value
            energy = np.logspace(np.log10(vals.min()), np.log10(vals.max()), 100)
            energy = Quantity(energy, self.ebounds.unit)

        aeff = self.evaluate(offset, energy).T
        extent = [
            offset.value.min(), offset.value.max(),
            energy.value.min(), energy.value.max(),
        ]
        ax.imshow(aeff.value, extent=extent, **kwargs)

        ax.set_yscale('log')
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
        ss += "-----------------------------\n"
        # Summarise data members
        ss += array_stats_str(self.ebounds, 'Energy')
        ss += array_stats_str(self.offset, 'Offset')
        ss += array_stats_str(self.eff_area, 'Effective Area')
        ss += 'Safe energy threshold lo: {0:6.3f}\n'.format(self.low_threshold)
        ss += 'Safe energy threshold hi: {0:6.3f}\n'.format(self.high_threshold)

        offset = Angle(0.5, 'deg')
        energy = Energy(1, 'TeV')
        effarea = self.evaluate(offset=offset, energy=energy)
        ss += 'Effective area at {} and {} : {:.3f}'.format(
            offset, energy, effarea)

        return ss

    def _prepare_linear_interpolator(self, interp_kwargs):
        from scipy.interpolate import RegularGridInterpolator

        x = self.offset.value
        y = np.log10(self.ebounds.log_centers.value)
        points = (x, y)
        values = self.eff_area.value

        self._linear = RegularGridInterpolator(points, values, **interp_kwargs)

    def _prepare_spline_interpolator(self):
        """Only works for radial symmetric input files (N=2)
        """

        # TODO Replace by scipy.ndimage.interpolation.map_coordinates

        from scipy.interpolate import RectBivariateSpline

        x = self.offset.value
        y = np.log10(self.ebounds.log_centers.value)

        self._spline = RectBivariateSpline(x, y, self.eff_area.value)


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
