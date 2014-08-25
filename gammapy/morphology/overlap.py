# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Source overlap tools.
"""
from __future__ import print_function, division
import json
import os
import logging
import numpy as np
from numpy import exp, pi, log, sqrt
from astropy.extern.configobj import configobj
from astropy.io import fits, ascii
from astropy.table import Table, Column
from astropy.wcs import WCS

# TODO: add what is publicly used here
__all__ = ['read_model_components']


def write_test_cfg_file(spacing=2.):
    """Write test fit.cfg file with 7 sources on a hexagonal grid.

    Please specify sigma, excess in fit.cfg and spacing of the sources.
    """
    cfg = configobj.ConfigObj('fit.cfg')

    # Hexgonal test grid
    positions = np.array([[0, 0], [0, 1], [1, 0.5], [1, -0.5], [0, -1], [-1, -0.5], [-1, 0.5]]) * spacing 

    # Set source parameters
    for i, pos in enumerate(positions):
        cfg['Source{0}'.format(i)] = {}
        cfg['Source{0}'.format(i)]['Type'] = 'Gaussian'
        cfg['Source{0}'.format(i)]['GLON'] = pos[0] + 180
        cfg['Source{0}'.format(i)]['GLAT'] = pos[1]
        cfg['Source{0}'.format(i)]['Sigma'] = 1.
        cfg['Source{0}'.format(i)]['Norm'] = 1.
    cfg.write()


def write_test_region_file(component_table):
    """Write ds9 region file with source names and R80 containment radii.
    """
    reg_file = open('fit.reg', 'w')
    for row, component in enumerate(component_table['Name']):
        glon_pos = component_table['GLON'][row]
        glat_pos = component_table['GLAT'][row]
        sigma = component_table['Sigma'][row]
        r_containment = sqrt(2 * log(1 / (1 - 0.8))) * sigma
        reg_string = ('galactic;circle({0},{1},{2}) # color={{green}} width=2 text={{{3}}}'
                      ''.format(glon_pos, glat_pos, r_containment, component))
        reg_file.write(reg_string + '\n')
    reg_file.close()


def write_region_file(q_table, c_table, component_table):
    """Write region file to display Q and C factors in ds9.
    """
    pass  # Would be nice but actually not necessary


def write_test_model_components(component_table):
    """Write test model components fits.
    """
    if not os.path.exists('model_components'):
        os.mkdir('model_components')

    def gauss2d(x, y, xpos, ypos, excess, sigma):
        """A 2D Gaussian with integral given by the excess parameter"""
        r2 = (x - xpos) ** 2 + (y - ypos) ** 2
        exponent = -r2 / (2 * sigma ** 2)
        amplitude = excess  # / (2 * pi * sigma ** 2) #What makes sense here?
        return amplitude * exp(exponent)

    # Set up fits header
    w = WCS(naxis=2)
    w.wcs.crpix = [200, 200]
    w.wcs.cdelt = np.array([-0.02, 0.02])
    w.wcs.crval = [180, 0]
    w.wcs.ctype = ["GLON-CAR", "GLAT-CAR"]
    header = w.to_header()
    y, x = np.indices((401, 401))
    glon, glat = w.wcs_pix2world(x, y, 0)

    # All source components added
    all_components = np.zeros_like(x, dtype=np.float32)

    for row, component in enumerate(component_table['Name']):
        glon_pos = component_table['GLON'][row]
        glat_pos = component_table['GLAT'][row]
        sigma = component_table['Sigma'][row]
        norm = component_table['Norm'][row]
        data = gauss2d(glon, glat, glon_pos, glat_pos, norm, sigma)
        hdu = fits.PrimaryHDU(data, header)
        hdulist = fits.HDUList(hdu)
        hdulist.writeto('model_components/{0}.fits'.format(component), clobber=True)
        all_components += data

    # Write Fits file with all components
    hdu = fits.PrimaryHDU(all_components, header)
    hdulist = fits.HDUList(hdu)
    hdulist.writeto('model_components/all.fits', clobber=True)

    # Write dummy counts.fits just for the wcs header
    hdu = fits.PrimaryHDU(all_components, header)
    hdulist = fits.HDUList(hdu)
    hdulist.writeto('../counts.fits', clobber=True)


def read_model_components(cfg_file):
    """Read model components from ``model_components/*.fits`` and return
    a list of 2D component images with containment masks.
    """
    cfg = configobj.ConfigObj(cfg_file)
    column_names = ('Name', 'Type', 'GLON', 'GLAT', 'Sigma', 'Norm')
    column_types = ('S25', 'S25', np.float32, np.float32, np.float32, np.float32)
    component_table = Table(names=column_names, dtype=column_types)

    # Build data table
    for component in cfg.keys():
        type_ = cfg[component]['Type']
        glon = cfg[component]['GLON']
        glat = cfg[component]['GLAT']
        sigma = cfg[component]['Sigma']
        norm = cfg[component]['Norm']
        component_table.add_row([component, type_, glon, glat, sigma, norm])
    if os.path.exists('model_components/'):
        read_fits_files(component_table)
    else:
        logging.error('No model components found. Please reuse morph_fit.')
    if os.path.exists('fit.reg'):
        read_region_file(component_table)
    else:
        compute_containment_radius(component_table)
        logging.info('Computing containment radii')
    return component_table


def compute_containment_radius(component_table, frac=0.8):
    """Compute containment radius from sigma.
    """
    r_containment_list = []
    for sigma in component_table['sigma']:
        r_containment = sqrt(2 * log(1 / (1 - frac))) * sigma  # TODO: Has to be tested!
        r_containment_list.append(r_containment)

    component_table.add_column(Column(data=r_containment_list, name='Containment Radii R80'))


def read_region_file(component_table):
    """Read region file to get the containment radii.
    """
    import pyregion
    region_list = pyregion.open('fit.reg')
    r_containment_list = []
    for component in region_list:
        r_containment = np.float(component.coord_list[2])
        r_containment_list.append(r_containment)
    component_table.add_column(Column(data=r_containment_list, name='Containment Radii R80'))


def read_fits_files(component_table):
    """Read model component FITS files.
    """
    image_data = []
    for component in component_table['Name']:
        component_fits = component.replace(' ', '_').replace('-', 'm').replace('+', 'p') + '.fits'
        data = fits.getdata('model_components/' + component_fits)
        image_data.append(data)
    component_table.add_column(Column(data=image_data, name='ImageData'))


def get_containment_mask(glon_pos, glat_pos, r_containment, shape):
    """Get mask from pre-computed containment radius.
    """
    hdulist = fits.open('../counts.fits')
    w = WCS(hdulist[0].header)
    y, x = np.indices(shape)
    glon, glat = w.wcs_pix2world(x, y, 1)

    # Fix glon and glon_pos
    glon = np.select([glon > 180, glon <= 180], [glon - 360, glon])
    glon_pos = np.select([glon_pos > 180, glon_pos <= 180], [glon_pos - 360, glon_pos])

    # Compute containment radius
    mask = (glon - glon_pos) ** 2 + (glat - glat_pos) ** 2 <= r_containment ** 2
    return np.array(np.reshape(mask, shape), dtype=np.int)


def get_containment_mask_from_sigma(glon_pos, glat_pos, sigma, frac, shape):
    """Compute mask for the containment radius.

    Works only for a symmetric Gaussian.
    """
    # Setup wcs coordinate transformation
    hdulist = fits.open('../counts.fits')
    w = WCS(hdulist[0].header)
    y, x = np.indices(shape)
    glon, glat = w.wcs_pix2world(x, y, 1)

    # Compute containment radius
    r_containment = sqrt(2 * log(1 / (1 - frac))) * sigma  # Has to be tested!
    mask = (glon - glon_pos) ** 2 + (glat - glat_pos) ** 2 <= r_containment ** 2
    return np.array(np.reshape(mask, shape), dtype=np.int)


def compute_circular_masks(x_max, y_max, shape):
    """Compute a 3D array of circular masks.

    with increasing radius at position x_max, y_max of given shape.
    """
    # TODO: Not used ... remove?
    width, height = shape
    r_max = min(width, height) / 2.
    y, x = np.mgrid[-y_max:height - y_max, -x_max:width - x_max]
    circular_masks = []
    for radius in np.arange(1, r_max):
        mask = x ** 2 + y ** 2 <= radius ** 2
        circular_masks.append(mask)
    return circular_masks


def containment_fraction(A, frac):
    """Compute containment fraction and return the corresponding mask.

    Does not interpolate.
    """
    # Not used
    y_max, x_max = np.unravel_index(np.argmax(A), A.shape)
    A_norm = A / np.nansum(A)
    circular_masks = compute_circular_masks(x_max, y_max, A.shape)
    for mask in circular_masks:
        if np.nansum(A_norm[mask]) >= frac:
            return mask


def contamination(B, excess_all, mask_A):
    """Compute contamination between A and B.

    I.e. fraction of B in a given containment region around A.
    """
    # mask = np.ones_like(A) #test hack
    logging.debug('Excess of B in region A: {0}'.format(np.nansum((B * mask_A))))
    logging.debug('Total excess in region A: {0}'.format(np.nansum((excess_all * mask_A))))
    return np.nansum(B * mask_A) / np.nansum(excess_all * mask_A)
    

def Q_factor(A, B):
    """Compute the "overlap" between the images A and B.
    """
    A_norm = np.nansum(A ** 2) ** 0.5
    B_norm = np.nansum(B ** 2) ** 0.5
    values = (A / A_norm) * (B / B_norm)
    return np.nansum(values)


def Q_factor_analytical(sigma_A, sigma_B, x_A, y_A, x_B, y_B, sigma_PSF=0.0):
    """Compute overlap Q factor analytically by means of the given model parameters.
    """
    # Compute convolved sigma 
    sigma_A = sqrt(sigma_A ** 2 + sigma_PSF ** 2)
    sigma_B = sqrt(sigma_B ** 2 + sigma_PSF ** 2)

    # sigma_AB squared
    sigma_AB2 = sigma_A ** 2 + sigma_B ** 2

    # displacement x_AB squared
    x_AB2 = (x_A - x_B) ** 2 + (y_A - y_B) ** 2

    # Normalization constant
    N = 2. * sigma_A * sigma_B / sigma_AB2
    return N * exp(-0.5 * x_AB2 / sigma_AB2)


def gaussian_product_integral(parameters_A, parameters_B):
    """Computes the analytical result of \int A \cdot B d^2x integrated from -inf to inf.
    """
    x_A, y_A, sigma_A, N_A = parameters_A
    x_B, y_B, sigma_B, N_B = parameters_B

    # Sigma_AB squared
    sigma_AB2 = sigma_A ** 2 + sigma_B ** 2

    # Displacement x_AB squared
    x_AB2 = (x_A - x_B) ** 2 + (y_A - y_B) ** 2

    # Prefactor
    N = N_A * N_B * sigma_A ** 2 * sigma_B ** 2 / sigma_AB2
    return N * exp(-0.5 * x_AB2 / sigma_AB2)


def compute_Q_matrix(components_A, components_B):
    """Compute matrix of pairwise Q_factors weighted with N and sigma.
    """
    Q_AB_matrix = np.empty((len(components_A), len(components_B)))
    for i, A in enumerate(components_A):
        for j, B in enumerate(components_B):
            Q_AB_matrix[i][j] = gaussian_product_integral(A, B)
    return Q_AB_matrix


def compute_Q_from_components(components_A, components_B):
    """Compute Q factor for sources A and B with several components.
    """
    Q_AB_matrix = compute_Q_matrix(components_A, components_B)
    Q_AA_matrix = compute_Q_matrix(components_A, components_A)
    Q_BB_matrix = compute_Q_matrix(components_B, components_B)
    return Q_AB_matrix.sum() / (sqrt(Q_AA_matrix.sum()) * sqrt(Q_BB_matrix.sum()))


def apply_PSF(components):
    """Given a list of Gaussian input components the PSF is applied analytically."""
    # Does not work yet, but could be useful
    sigma_PSF_1, N_PSF_1 = 1., 1.
    sigma_PSF_2, N_PSF_2 = 1., 1.
    sigma_PSF_3, N_PSF_3 = 1., 1.

    components_PSF_applied = []
    for component in components:
        x, y, sigma, N = component

        sigma_1 = sqrt(sigma ** 2 + sigma_PSF_1 ** 2)
        sigma_2 = sqrt(sigma ** 2 + sigma_PSF_2 ** 2)
        sigma_3 = sqrt(sigma ** 2 + sigma_PSF_3 ** 2)
        N_1 = 2 * pi * N * N_PSF_1 * sigma_PSF_1 ** 2 * sigma ** 2 / sigma_1 ** 2
        N_2 = 2 * pi * N * N_PSF_2 * sigma_PSF_2 ** 2 * sigma ** 2 / sigma_2 ** 2
        N_3 = 2 * pi * N * N_PSF_3 * sigma_PSF_3 ** 2 * sigma ** 2 / sigma_3 ** 2
        components_PSF_applied.append([x, y, sigma_1, N_1])
        components_PSF_applied.append([x, y, sigma_2, N_2])
        components_PSF_applied.append([x, y, sigma_3, N_3])
    return components_PSF_applied


def compute_Q_analytical(component_table):
    """Compute Q factors analytically.
    """
    q_table = Table()
    q_table.add_column(Column(data=component_table['Name'], name='Q_AB'))

    Q_all_list = ['All others']

    for j in range(len(component_table)):
        # Get parameters A
        row_A = component_table[j]
        x_A, y_A, sigma_A, N_A = row_A['GLON'], row_A['GLAT'], row_A['Sigma'], row_A['Norm']

        # Compute Q_factor all others
        components_all = [[row['GLON'], row['GLAT'], row['Sigma'], row['Norm']] for row in component_table if not row == row_A]
        Q_All = compute_Q_from_components([[x_A, y_A, sigma_A, N_A]], components_all)
        Q_all_list.append(Q_All)

        # Compute Q factors pairwise
        Q_AB_list = np.zeros(len(component_table))
        for i, row_B in enumerate(component_table):
            # Get parameters B
            x_B, y_B, sigma_B, N_B = row_B['GLON'], row_B['GLAT'], row_B['Sigma'], row_B['Norm']

            # Compute Q_factor
            Q_AB = Q_factor_analytical(sigma_A, sigma_B, x_A, y_A, x_B, y_B)
            Q_AB_list[i] = Q_AB
        q_table.add_column(Column(data=Q_AB_list, name=row_A['Name']))
    q_table.add_row(Q_all_list)
    return q_table


def compute_Q(component_table):
    """Compute Q factors.
    """
    q_table = Table()
    q_table.add_column(Column(data=component_table['Name'], name='Q_AB'))

    Q_all_list = ['All others']
    excess_all = np.nansum(component_table['ImageData'], axis=0)  # Total excess

    for row_A in range(len(component_table)):
        # Compute q factors all others
        Q_All = Q_factor(component_table['ImageData'][row_A], excess_all - component_table['ImageData'][row_A])
        Q_all_list.append(Q_All)

        # Compute Q factors pairwise
        Q_AB_list = np.zeros(len(component_table))
        for row_B in range(len(component_table)):
            Q_AB = Q_factor(component_table['ImageData'][row_A], component_table['ImageData'][row_B])
            Q_AB_list[row_B] = Q_AB
        q_table.add_column(Column(data=Q_AB_list, name=component_table['Name'][row_A]))
    q_table.add_row(Q_all_list)
    return q_table


def compute_contamination(component_table, store_mask=False):
    """Compute contamination.
    """
    q_table = Table()
    q_table.add_column(Column(data=component_table['Name'], name='Contamination'), 0)

    C_all_list = ['All others']
    excess_all = np.nansum(component_table['ImageData'], axis=0)
    # Compute contamination factors
    for row_A in range(len(component_table)):
        # Get source component parameters for containment mask
        glon_pos = component_table['GLON'][row_A]
        glat_pos = component_table['GLAT'][row_A]
        # sigma = component_table['Sigma'][row_A]    
        shape = component_table['ImageData'][row_A].shape
        r_containment = component_table['Containment Radii R80'][row_A]
        containment_mask_A = get_containment_mask(glon_pos, glat_pos, r_containment, shape)


        # Store mask for debugging
        if store_mask:
            hdu = fits.PrimaryHDU(containment_mask_A)
            hdulist = fits.HDUList([hdu])
            name = component_table['Name'][row_A].replace(' ', '_').replace('-', 'm').replace('+', 'p')
            hdulist.writeto('model_components/mask_{0}.fits'.format(name), clobber=True)

        # Compute contamination all others
        C_All = contamination(excess_all - component_table['ImageData'][row_A], excess_all, containment_mask_A)
        C_all_list.append(C_All)

        # Compute pairwise contamination
        C_list = np.zeros(len(component_table))
        for row_B in range(len(component_table)):
            C_AB = contamination(component_table['ImageData'][row_B], excess_all, containment_mask_A)
            C_list[row_B] = C_AB
        q_table.add_column(Column(data=C_list, name=component_table['Name'][row_A]))
    q_table.add_row(C_all_list)
    return q_table


def write_json_fit(q_table, filename):
    """Write Q and C factors in fit.json file.
    """
    data = json.load(open(filename))

    # Check if key "overlap" is already there
    if 'overlap' not in data.keys():
        data['overlap'] = dict()

    # Write table to json file
    overlap, columns_names = q_table.colnames[0], q_table.colnames[1:]
    data['overlap'][overlap] = dict()
    for source_1 in columns_names:
        data['overlap'][overlap][source_1] = dict()
        for row in range(len(q_table[overlap])):
            data['overlap'][overlap][source_1][q_table[overlap][row]] = q_table[source_1][row] 

    json.dump(data, open(filename, 'w'),
                      sort_keys=True, indent=4)


def write_json(q_table, filename='overlap.json'):
    """Write Q and C factors in overlap.json file.
    """
    if not os.path.exists(filename):
        data = dict()
    else:
        data = json.load(open(filename))

    # Write table to json file
    overlap, columns_names = q_table.colnames[0], q_table.colnames[1:]
    data[overlap] = dict() 
    for source_1 in columns_names:
        data[overlap][source_1] = dict()
        for row in range(len(q_table[overlap])):
            data[overlap][source_1][q_table[overlap][row]] = q_table[source_1][row] 

    json.dump(data, open(filename, 'w'),
                      sort_keys=True, indent=4)

def write_pretty_table(table, filename, column_width=25):
    """Format and write pretty table to ascii file.
    """
    formats = {}

    # Format header
    colnames = ['{0:>{width}} '.format(name, width=column_width) for name in table.colnames]

    # Format columns
    for i, column in enumerate(colnames):
        if table.columns[i].dtype == '>S25':  # Strings have to be stored in S25 format! Otherwise the formatting will be wrong...
            formats[column] = '{:>' + str(column_width) + '} '
        else:
            formats[column] = '{:>' + str(column_width) + '.2%} '

    # Write table
    table.write(filename, format='ascii', Writer=ascii.FixedWidthTwoLine, delimiter='|',
                                    position_char='=', formats=formats, names=colnames)
    logging.info('Wrote table to file: {0}'.format(filename))


def compare_Q_contamination(component_table):
    """Compare the values of the Q factor and contamination.
    """
    # Q Factors
    q_table = compute_Q(component_table)
    write_pretty_table(q_table, 'q_factors.txt')
    write_json(q_table)

    # Q Factors analytical
    # q_table = compute_Q_analytical(component_table)
    # write_pretty_table(q_table, 'q_factors_analytical.txt')

    # Contamination
    c_table = compute_contamination(component_table)
    write_pretty_table(c_table, 'contamination.txt')
    write_json(c_table)


def compare_real_and_model_excess(component_table):
    """Compare real and model excess by calculating the ratio.
    """
    # Read fits files
    roi = fits.getdata('../roi.fits')
    counts = fits.getdata('../counts.fits')
    background = fits.getdata('../background.fits')
    real_excess = counts - background
    model_excess = fits.getdata('model.fits')

    # Compute total excess
    total_real_excess = np.nansum(roi * real_excess)
    total_model_excess = np.nansum(roi * model_excess)
    logging.info('Total model excess: {0:02.2f}'.format(total_model_excess))
    logging.info('Total real excess: {0:02.2f}'.format(total_real_excess))
    logging.info('Ratio: {0:02.2f}'.format(total_model_excess / total_real_excess))


def overlap_plot(profile_A, profile_B, w, y_slice):
    """Make nice overlap plots.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplots_adjust(hspace=0.001)

    norm_product_AB = profile_A * profile_B
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax1.plot(profile_A, label='Source A', linewidth=4, color='b')
    ax1.plot(profile_B, label='Source B', linewidth=4, color='g')
    ax1.tick_params(axis='both', which='major', labelsize=16)
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.legend()

    ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    ax2.plot(norm_product_AB, label='Product', linewidth=4, color='r')
    plt.yticks(np.arange(0, 0.5, 0.25))
    plt.ylim(0, 0.5)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    pos = np.linspace(0, profile_A.size, 11)
    y_labels, x_labels = w.wcs_pix2world(pos, np.ones_like(pos) * y_slice, 0)
    plt.xticks(pos, x_labels)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.legend()
    # plt.show()


def make_example_plots(component_table):
    """Make some example plots from a `fake.cfg`.
    """
    import matplotlib.pyplot as plt
    hdulist = fits.open('../exposure.fits')

    # Parse the WCS keywords in the primary HDU
    w = WCS(hdulist[0].header)

    # Slice through center
    y_slice, x_slice = w.wcs_world2pix(0, 0, 0)

    for row_A in range(len(component_table)):
        for row_B in range(len(component_table)):
            profile_A = component_table['ImageData'][row_A][y_slice][:]
            profile_B = component_table['ImageData'][row_B][y_slice][:]
            overlap_plot(profile_A, profile_B, w, y_slice)
            A = component_table['Name'][row_A]
            B = component_table['Name'][row_B]
            plt.savefig("OverlapPlot{0}{1}.png".format(A, B), dpi=160)
