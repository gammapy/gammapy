# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Helper functions for working with GALPROP results.

prepare(), reproject_to() and make_mask_and_area() should
be run to prepare the results in a nice format.

Then use the Galprop class to compute spectra, profiles and plot them.

See process_galprop for an example of how to use this module.

TODO:
- correct calculation of integral flux map for given energy band (interpolate and sum)
- add methods to compare with HESS data
- Check out where the following warning occurs:
Warning: divide by zero encountered in log"""
from os.path import join
import logging
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from kapteyn.maputils import FITSimage
import atpy
import image.utils as iu
from spec.powerlaw import f_from_points, I_from_points, g_from_points

# Component lists. An OrderedDict would be nicer.
components = ['total', 'pion_decay', 'ics_isotropic', 'bremss']
names = ['Total', 'Pion Decay', 'Inverse Compton', 'Bremsstrahlung']
colors = ['black', 'red', 'green', 'cyan']


def cube_integrate(fluxes, energies, emin=1e6, emax=1e20):
    """Calculate integral flux by summing integral fluxes of bins,
    assuming the spectrum is a power law in each bin."""
    # Find bin indices i_lo, i_hi such that
    # emin and emax are contained in the bins
    indices = np.where((energies >= emin) & (energies <= emax))[0]
    i_lo, i_hi = indices[0], indices[-1]
    # Find energy and differential flux at each bin edge.
    # Note that + 1 is required to read the values at the right edge.
    energies = energies[i_lo: i_hi + 1]
    fluxes = fluxes[i_lo: i_hi + 1]
    # Now fix the first and last bin to exactly match the
    # energy range selected by the user
    energies[0] = emin
    fluxes[0] = f_from_points(energies[0], energies[1],
                             fluxes[0], fluxes[1], emin)
    energies[-1] = emax
    fluxes[-1] = f_from_points(energies[-2], energies[-1],
                               fluxes[-2], fluxes[-1], emax)
    # Now compute the integral flux per bin
    int_fluxes = I_from_points(energies[:-1], energies[1:],
                               fluxes[:-1], fluxes[1:])
    logging.debug('energies = {0}'.format(energies))
    logging.debug('fluxes = {0}'.format(fluxes))
    logging.debug('int_fluxes = {0}'.format(int_fluxes))
    return int_fluxes.sum()


def filename(tag, ii=0, filetype='cube'):
    return '_'.join([tag, filetype, components[ii] + '.fits'])


def prepare(galprop_dir, tag='orig', clobber=True):
    """Convert GALPROP output files to a format I like.

    1) Put the Galactic center to the image center, instead
       of having the Galactic anti-center at the image center.
    2) Have the longitude axis increase to the left
    3) Create a total cube, which is the sum of the three
       components.
    4) Use simple file names.

    The modified cubes are written to results_dir, which
    is then used by the GALPROP class."""
    def fix(hdu):
        # Get data and header of the cube
        data, header = hdu.data, hdu.header

        # Fix header
        header['CDELT1'] = -header['CDELT1']
        header['CRVAL1'] = 0

        # fix data
        half_naxis = header['NAXIS1'] / 2
        left = data[:, :, :half_naxis]
        right = data[:, :, half_naxis:]
        data = np.dstack((right, left))
        data = data[:, :, ::-1]

        # Store the changes
        hdu.data, hdu.header = data, header

    def map_id():
        _ = galprop_dir.split('/')
        dirname = _[-1] if _[-1] else _[-2]
        return '_'.join(dirname.split('_')[1:3])

    # Copy and fix all the components
    infiles = [join(galprop_dir, component + '_mapcube_' +
                    map_id() + '.gz') for component in components]
    outfiles = [filename(tag, ii) for ii in range(len(components))]
    for ii in [1, 2, 3]:  # Total component doesn't exist yet
        logging.info('Fixing {0}'.format(components[ii]))
        hdulist = fits.open(infiles[ii])
        fix(hdulist['PRIMARY'])
        hdulist.writeto(outfiles[ii], clobber=clobber)
    logging.info('Computing total cube')
    total = fits.open(outfiles[1])
    for ii in [2, 3]:
        total['PRIMARY'].data += fits.getdata(outfiles[ii])
    total.writeto(outfiles[0], clobber=clobber)


def reproject_to(ref_file, in_tag='orig', out_tag='repro',
                 interpol_dict={'order': 3},
                 clobber=True):
    """Reproject cubes e.g. to match a HESS survey map.

    Can be computationally and memory intensive if the
    HESS map is large or you have many energy bands."""
    logging.info('Reprojecting to {0}'.format(ref_file))
    ref_image = FITSimage(ref_file)
    for ii in range(len(components)):
        in_file = filename(in_tag, ii)
        out_file = filename(out_tag, ii)
        logging.info('Reading {0}'.format(in_file))
        in_image = FITSimage(in_file)
        logging.info('Reprojecting ...')
        out_image = in_image.reproject_to(ref_image.hdr,
                                          interpol_dict=interpol_dict)
        logging.info('Writing {0}'.format(out_file))
        out_image.writetofits(out_file, clobber=clobber)

        # TODO: do this with `astropy.io.fits` instead of calling `fappend`!
        logging.info('Copying energy extension')
        cmd = ['fappend',
               '{0}[ENERGIES]'.format(in_file),
               '{0}'.format(out_file)]
        #call(cmd)
        raise NotImplementedError


def make_mask_and_area(ref_file, clobber=True):
    """Make the mask and area images for use by Galprop
    mask == 0 pixels are excluded,
    mask == 1 pixels are included"""
    ref_image = FITSimage(ref_file)
    mask = (ref_image.dat != 0).astype(np.uint8)
    mask_image = FITSimage(externaldata=mask,
                           externalheader=ref_image.hdr)
    mask_image.writetofits('mask.fits', clobber=clobber)
    area = iu.area(ref_image, deg=False)
    area_image = FITSimage(externaldata=area,
                           externalheader=ref_image.hdr)
    area_image.writetofits('area.fits', clobber=clobber)


class Galprop:
    """Compute and plot spectra and projections of
    arbitrary regions, defined by a mask."""
    def __init__(self, tag='repro', clobber=False):
        """dir should contain the cubes already as produced by prepare()"""
        self.tag = tag
        self.clobber = clobber
        # Use toal component as reference
        self.ref_file = filename(tag=tag)
        self.fitsimage = FITSimage(self.ref_file)
        # Construct vectors of glon, glat, energy e.g. for plotting
        ac = iu.axis_coordinates(self.fitsimage)
        self.glon = ac['GLON']
        self.glat = ac['GLAT']
        self.energy = 10 ** ac['PHOTON ENERGY']
        # Read mask if there is one, else don't use a mask
        try:
            self.mask = fits.getdata('mask.fits')
            logging.info('Loaded mask.fits')
        except IOError:
            self.mask = 1
            logging.info('mask.fits not found')
        try:
            self.area = fits.getdata('area.fits')
            logging.info('Loaded area.fits')
        except IOError:
            self.area = iu.area(self.fitsimage, deg=False)
            logging.info('area.fits not found')

    def calc_spec(self):
        """Calculate spectra"""
        logging.info('---> Calculating spectra')
        table = atpy.Table()
        table.add_column('energy', self.energy)
        for ii in range(len(components)):
            in_file = filename(self.tag, ii)
            sb = fits.getdata(in_file)
            # Compute average spectrum as total flux / total area
            flux = self.mask * self.area * sb
            area = self.mask * self.area
            # Note that the correct way to average two pixels with
            # areas a1 and a2 and surface brightnesses sb1 and sb2 is:
            # mean_sb = (a1 * sb1 + a2 * sb2) / (a1 + a2)
            flux_sum = flux.sum(-1).sum(-1)
            area_sum = area.sum(-1).sum(-1)
            sb_mean = flux_sum / area_sum
            spec = self.energy ** 2 * sb_mean
            colname = components[ii]
            table.add_column(colname, spec)
        table.write('spec.fits', overwrite=self.clobber)

    def calc_spec_index(self):
        """Calculate spectral indices"""
        # Note that we can't compute a forward difference
        # of the rightmost flux value, so the table is one shorter.
        logging.info('---> Calculating spectral indices')
        in_table = atpy.Table('spec.fits')
        energy = in_table.data['energy']
        out_table = atpy.Table(name='Spectral Index')
        out_table.add_column('energy', energy[:-1])
        for ii in range(len(components)):
            flux = in_table.data[components[ii]]
            spec_index = g_from_points(energy[:-1], energy[1:],
                                       flux[:-1], flux[1:])
            spec_index += 2  # Because we stored E**2 dN / dE
            out_table.add_column(components[ii], spec_index)
        out_table.write('spec_index.fits', overwrite=self.clobber)

    def make_int_flux_image(self, emin=1e6, emax=1e20):
        """Make integral flux for an energy band"""
        for ii in range(len(components)):
            in_file = filename(self.tag, ii, 'cube')
            out_file = filename(self.tag, ii, 'image')
            logging.info('---> Processing {0}'.format(components[ii]))
            fluxes = fits.getdata(in_file)
            image = cube_integrate(fluxes, self.energy, emin, emax)
            fits.writeto(out_file, image, clobber=self.clobber)

    def make_diff_flux_image(self, energy=1e6):
        """Make differential flux image at a given energy"""
        for ii in range(len(components)):
            in_file = filename(self.tag, ii, 'cube')
            out_file = filename(self.tag, ii, 'image')
            slicepos = np.where(self.energy >= energy)[0][0]
            logging.debug('Selecting slice {0} at energy {1}'
                         ''.format(slicepos, self.energy[slicepos]))
            hdu = fits.open(in_file)[0]
            hdu = iu.cube_to_image(hdu, slicepos)
            logging.info('Writing {0}'.format(out_file))
            hdu.writeto(out_file, clobber=self.clobber)

    def calc_profiles(self):
        """Calculate latitude and longitude profiles"""
        logging.info('---> Calculating profiles')
        for axis in ['GLON', 'GLAT']:
            if axis == 'GLON':
                pos = self.glon
                axis_index = 0
            else:
                pos = self.glat
                axis_index = 1
            table = atpy.Table()
            table.add_column(axis, pos)
            for ii in range(len(components)):
                in_file = filename(self.tag, ii, 'image')
                sb = fits.getdata(in_file)
                # Compute average surface brightness
                flux = self.mask * self.area * sb
                area = self.mask * self.area
                flux_sum = flux.sum(axis_index)
                area_sum = area.sum(axis_index)
                mean_sb = flux_sum / area_sum
                table.add_column(components[ii], mean_sb)
            out_file = '{0}_profiles.fits'.format(axis)
            logging.info('Writing {0}'.format(out_file))
            table.write(out_file, overwrite=self.clobber)

    def _plot(self, in_file,
              xcolname, xlabel, ylabel):
        """Plot all components"""
        plt.clf()
        table = atpy.Table(in_file)
        x = table.data[xcolname]
        for ii in range(len(components)):
            y = table.data[components[ii]]
            plt.plot(x, y, color=colors[ii], label=names[ii])
        plt.grid()
        plt.legend(loc='best')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def plot_spec(self):
        """Plot spectra"""
        in_file = 'spec.fits'
        out_file = 'spec.pdf'
        xcolname = 'energy'
        xlabel = 'Photon Energy (MeV)'
        ylabel = 'Flux E^2 F (MeV cm^-2 sr^-1 s^-1)'
        self._plot(in_file, xcolname, xlabel, ylabel)
        plt.loglog()
        plt.savefig(out_file)

    def plot_spec_index(self):
        """Plot spectral indices"""
        in_file = 'spec_index.fits'
        out_file = 'spec_index.pdf'
        xcolname = 'energy'
        xlabel = 'Photon Energy (MeV)'
        ylabel = 'Spectral Index'
        self._plot(in_file, xcolname, xlabel, ylabel)
        plt.ylim(2, 4)
        plt.semilogx()
        plt.savefig(out_file)

    def plot_profile(self, axis='GLAT'):
        """Plot average GLAT or GLON profile at a given energy"""
        logging.info('---> Plotting profiles')
        in_file = '{0}_profiles.fits'.format(axis)
        out_file = '{0}_profiles.pdf'.format(axis)
        xcolname = axis
        xlabel = '{0} (deg)'.format(axis)
        ylabel = 'Average Flux E^2 F (MeV cm^-2 sr^-1 s^-1)'
        self._plot(in_file, xcolname, xlabel, ylabel)
        plt.semilogy()
        plt.savefig(out_file)

    def do_all(self):
        """Process all common steps"""
        # Spectra
        self.calc_spec()
        self.calc_spec_index()
        self.plot_spec()
        self.plot_spec_index()
        # Image
        # self.make_int_flux_image()
        self.make_diff_flux_image()
        # Profiles
        self.calc_profiles()
        self.plot_profile('GLON')
        self.plot_profile('GLAT')


def main():
    logging.basicConfig(level=logging.DEBUG)

    ################################
    # Directory and file names
    ################################
    base_dir = '/nfs/d22/hfm/hillert/data-files/galprop'
    run = 'results_54_0353000q_zdguc6ruot1bue7f'
    galprop_dir = join(base_dir, run)
    results_dir = join(galprop_dir, 'results')
    ref_file = join(base_dir, 'hess_exclusion_0.3_part.fits')

    # Create and set result dir
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    os.chdir(results_dir)

    ################################
    # Prepare data
    ################################

    galprop.prepare(galprop_dir)
    galprop.reproject_to(ref_file)
    galprop.make_mask_and_area(ref_file)

    ################################
    # Compute results and make plots
    ################################

    g = galprop.Galprop(clobber=True)
    g.do_all()

    #g.calc_spec()
    #g.calc_spec_index()
    #g.plot_spec()
    #g.plot_spec_index()

    #g.make_int_flux_image()
    #g.make_diff_flux_image()
    #g.calc_profiles()
    #g.plot_profile('GLAT')
    #g.plot_profile('GLON')


if __name__ == '__main__':
    main()
