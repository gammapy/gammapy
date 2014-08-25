# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Interface to the Fermi and HESS catalogs.
"""
from __future__ import print_function, division
import logging
import numpy as np
from astropy.table import Table, Column
from astropy.units import Unit, Quantity
from ..spectrum import compute_differential_flux_points

__all__ = ['SEDComponent',
           'SED',
           'cube_sed',
           'add_spec',
           ]

MeV_to_GeV = Unit('MeV').to(Unit('GeV'))
MeV_to_erg = Unit('MeV').to(Unit('erg'))


class SEDComponent(object):
    """Uniform interface to SED components for the SED class
    """
    def __init__(self, name='', model=None, points=None):
        """
        @param name: str
        @type model: spec.spectrum.Spectrum
        @type points: spec.data.FluxPoints"""
        self.name = name
        self.model = model
        self.points = points

    def plot(self, model=True, points=True, butterfly=True):
        if butterfly:
            self.plot_butterfly()
        if model:
            self.plot_model()
        if points:
            self.plot_points()

    def plot_model(self):
        import matplotlib.pyplot as plt
        if self.model is None:
            logging.warning('{0}: No model available.'.format(self.name))
            return
        x, y = self.model.points(power=2)
        plt.plot(x * MeV_to_GeV, y * MeV_to_erg, label=self.name)

    def plot_points(self, color='black', markerfacecolor='black'):
        import matplotlib.pyplot as plt
        if self.points is None:
            logging.warning('{0}: No points available.'.format(self.name))
            return
        # @note We plot each point individually because anyway
        # upper limits have to be plotted differently which I
        # think is not possible because the marker argument doesn't
        # take arrays.
        for ii in range(len(self.points)):
            x, exl, exh, y, eyl, eyh, ul = self.points[ii]
            if ul:
                marker = ' '
                lolims = True
                eyl, eyh = 5 / 10. * y, 0
            else:
                marker = 'o'
                lolims = False
            plt.plot(MeV_to_GeV * x, y)
            plt.errorbar(x, y, [eyl, eyh], [exl, exh], lolims=lolims,
                         marker=marker, color=color,
                         markerfacecolor=markerfacecolor,
                         label=self.name)

    def plot_butterfly(self):
        pass


class SED(list):
    """Class to plot GeV -- TeV SEDs

    Internally the same units as in the Fermi catalog are used:
    - Energies in MeV
    - Flux densities in cm^-2 s^-2 MeV^-1
    - Fluxes in cm^-2 s^-1
    - Energy fluxes in erg cm^-2 s^-1"""
    """
    def add_Fermi(self, name):
        try:
            self._fermi
            self.append(self._fermi.sed_component(name))
        except
        component = catalog.sed_component(name)
        self.append(component)
    """
    def add(self, names, catalogs):
        for name in names:
            for catalog in catalogs:
                try:
                    component = catalog.sed_component(name)
                    self.append(component)
                    logging.info('%s found in %s',
                                 name, catalog.table.table_name)
                except ValueError as e:
                    logging.warning(e)
                    logging.warning('%s not found in %s',
                                    name, catalog.table.table_name)
                    pass

    def plot(self, filename='sed.png', xlim=(8e-2, 2e5), ylim=(1e-14, 1e-8)):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.ylabel(r'E$^2$ dF/DE (erg cm$^{-2}$ s$^{-1}$)')
        plt.xlabel('Energy (GeV)')
        plt.loglog()
        logging.info('Plotting {0} components in SED'.format(len(self)))
        for component in self:
            component.plot()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        logging.info('Writing {0}'.format(filename))
        plt.savefig(filename)

    def add_component(self, catalog_format, catalog_name,
                      object_name, plot_pivot=False, **ecpl_params):
        """ Read necessary parameters from FITS file and plot butterfly

        Parameters:
        catalog_format = 'hess', 'fermi'
        catalog_name = FITS file name
        object_name  = object name string in 'name' column

        Note: Since every catalog has columns with different
        names and units, a general SED plotting is not possible.
        Instead for each catalog type a handler function that
        deals converts to a standard format is called.

        @todo: Possibly pass plotting parameters along here by
        appending them to the ecpl_params dictionary
        -> I don't think this works at the moment!!!"""
        from atpy import Table
        # Get the catalog from file and initialize some things
        self.catalog_format = catalog_format
        self.catalog_name = catalog_name
        self.object_name = object_name
        self.catalog = Table(catalog_name).data
        # Build a dictionary of parameters needed for the plot
        self.ecpl_params = ecpl_params
        self.get_ecpl_params()
        # Plot curve
        self.plot_ecpl(plot_pivot=plot_pivot, **ecpl_params)
        # Plot points if present
        if self.plot_points is not None:
            # Get the values needed for plotting
            e = self.plot_points[0]
            f = self.plot_points[1]
            f_err = self.plot_points[2]
            e_err = self.plot_points[3]
            is_ul = self.plot_points[4]
            for ii in range(e.size):
                self.plot_point(e[ii], f[ii],
                                f_err=f_err[ii],
                                e_err=[[e_err[0][ii]], [e_err[1][ii]]],
                                ul=is_ul[ii])
            # Remove so that it doesn't get plotted again.
            self.plot_points = None

    def get_ecpl_params(self):
        """Build self.ecpl_params dictionary
        by parsing one of the supported catalogs"""
        if self.catalog_format == 'hess':
            self.get_ecpl_params_hess_cat()
        elif self.catalog_format == 'fermi':
            self.get_ecpl_params_fermi_cat()
        # Change numpy types to regular types
        # and replace nan values with 0
        for key, value in self.ecpl_params.items():
            if isinstance(value, np.float32):
                value = float(value)
            if isinstance(value, np.int16):
                value = int(value)

    def get_ecpl_params_fermi_cat(self):
        """ Build self.ecpl_params dictionary from Fermi catalog fields """
        i = self.find_object_index('source_name')
        # Set all plot parameters:
        self.ecpl_params['e_pivot'] = self.catalog.field('Pivot_Energy')[i]
        self.ecpl_params['e_min'] = 1e2
        self.ecpl_params['e_max'] = 1e5
        self.ecpl_params['e_cut'] = 0.0
        self.ecpl_params['e_cut_err'] = 0.0
        self.ecpl_params['e_scale'] = 1
        self.ecpl_params['norm'] = self.catalog.field('Flux_Density')[i]
        self.ecpl_params['norm_err'] = self.catalog.field('Unc_Flux_Density')[i]
        self.ecpl_params['norm_scale'] = 1
        self.ecpl_params['index'] = self.catalog.field('Spectral_Index')[i]
        self.ecpl_params['index_err'] = self.catalog.field('Unc_Spectral_Index')[i]
        self.ecpl_params['color'] = 'green'
        self.ecpl_params['butterfly'] = True
        # Set flux point data
        self.plot_points = self.get_flux_points_fermi(i)
        # Add text label
        fmt = '%s\n%s, %s\n' + \
            r'S = %3.1f, C = %3.1f, $\Gamma = %1.2f \pm %1.2f$'
        values = (self.object_name,
                  self.catalog.field('class1')[i],
                  self.catalog.field('assoc1')[i],
                  self.catalog.field('signif_avg')[i],
                  self.catalog.field('curvature_index')[i],
                  self.catalog.field('spectral_index')[i],
                  self.catalog.field('unc_spectral_index')[i]
                  )
        self.ax.text(0.05, 0.95, fmt % values,
                     horizontalalignment='left',
                     verticalalignment='top',
                     transform=self.ax.transAxes)


def add_spec(frame, model, xlim, npoints=100, **plot_params):
    """Add a spectral component to a frame.

    frame = matplotlib.Axes object
    model = [function, parameters, constants]
    xlim  = [xmin, xmax]"""
    # Unpack model
    f, p, c = model
    # Compute x and y values
    logx = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), npoints)
    x = 10 ** logx
    y = f(p, c, x)
    frame.plot(x, y, **plot_params)


def add_crab(ax):
    """Add the Fermi and HESS Crab SED to test scaling."""
    pass
    # The HESS butterfly
    # Note: The HESS catalog contains energies in TeV and flux norm in 1e-12 cm^-2 s^-1 TeV^-1
    """
    add_sed_component(ax, e0 = 1, e1 = 1e-2, e2 = 1e2,
                      norm = 10, norm_err = 0, index = 2., index_err = 0.0,
                      e_scale = 1e12, norm_scale = 1e-12 * 1e-12, e_cut = 10, e_cut_err = 3,
                      color='b', butterfly = True)
    # The Fermi butterfly
    add__sed_component(ax, e0 = 494, e1 = 1e2, e2 = 1e6,
                       norm = 1e-9, norm_err = 6.7e-11, index = 2.3, index_err = 0.1,
                       e_scale = 1e6, norm_scale = 1e-6, color='g', butterfly = True)
    # Add published fermi result
    """


def cube_sed(cube, mask=None, flux_type='differential', counts=None,
             errors=False, standard_error=0.1, spectral_index=2.3):
    """Creates SED from SpectralCube within given lat and lon range.

    Parameters
    ----------
    cube : `~gammapy.data.SpectralCube`
        Spectral cube of either differential or integral fluxes (specified
        with flux_type)
    mask : array_like, optional
        2D mask array, matching spatial dimensions of input cube.
        A mask value of True indicates a value that should be ignored, 
        while a mask value of False indicates a valid value.
    flux_type : {'differential', 'integral'}
        Specify whether input cube includes differential or integral fluxes.
    counts :  `~gammapy.data.SpectralCube`, optional
        Counts cube to allow Poisson errors to be calculated. If not provided,
        a standard_error should be provided, or zero errors will be returned.
    errors : bool
        If True, computes errors, if possible, according to provided inputs.
        If False (default), returns all errors as zero.
    standard_error : float
        If counts cube not provided, but error values required, this specifies
        a standard fractional error to be applied to values. Default = 0.1.
    spectral_index : float
        If integral flux is provided, this is used to calculate differential
        fluxes and energies (according to the Lafferty & Wyatt model-based
        method, assuming a power-law model).

    Returns
    -------
    table : `~astropy.table.Table`
        A spectral energy table of energies, differential fluxes and
        differential flux errors. Units as those input.
    """

    lon, lat = cube.spatial_coordinate_images

    values = []
    for i in np.arange(cube.data.shape[0]):
        if mask is None:
            bin = cube.data[i].sum()
        else:
            bin = cube.data[i][mask].sum()
        values.append(bin.value)
    values = np.array(values)

    if errors:
        if counts is None:
            # Counts cube required to calculate poisson errors
            errors = np.ones_like([values]) * standard_error
        else:
            errors = []
            for i in np.arange(counts.data.shape[0]):
                if mask is None:
                    bin = counts.data[i].sum()
                else:
                    bin = counts.data[i][mask].sum()
                r_error = 1. / (np.sqrt(bin.value))
                errors.append(r_error)
            errors = np.array([errors])
    else:
        errors = np.zeros_like([values])

    if flux_type == 'differential':
        energy = cube.energy
        table = Table()
        table['ENERGY'] = energy,
        table['DIFF_FLUX'] = Quantity(values, cube.data.unit),
        table['DIFF_FLUX_ERR'] = Quantity(errors * values, cube.data.unit)

    elif flux_type == 'integral':

        emins = cube.energy[:-1]
        emaxs = cube.energy[1:]
        table = compute_differential_flux_points(x_method='lafferty',
                                                 y_method='power_law',
                                                 spectral_index=spectral_index,
                                                 energy_min=emins, energy_max=emaxs,
                                                 int_flux=values, 
                                                 int_flux_err=errors * values)

    else:
        raise ValueError('Unknown flux_type: {0}'.format(flux_type))

    return table
