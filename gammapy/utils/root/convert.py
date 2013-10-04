# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility function TH2_to_FITS to export ROOT TH2 data to FITS files.

Run this file, then look at the FITS file:
$ ftlist TH2_to_FITS.fits H
$ ftlist TH2_to_FITS.fits K
$ ds9 TH2_to_FITS.fits
In the menu select:
> Analysis > Coordinate Grid
> WCS > Galactic & Degrees

@todo: Only use pyfits, not kapteyn.
@todo: Expand this into a small command line tool.
"""
from __future__ import print_function, division
import warnings
import numpy as np

__all__ = ['hist_to_table',
           'TH2_to_FITS_header', 'TH2_to_FITS_data', 'TH2_to_FITS']


def hist_to_table(hist):
    """Convert 1D ROOT histogram into astropy Table"""
    from rootpy import asrootpy
    from astropy.utils.compat.odict import OrderedDict
    from astropy.table import Table

    hist = asrootpy(hist)
    
    data = OrderedDict()
    data['x'] = list(hist.x())
    data['x_err'] = list(hist.xerravg())
    data['x_err_lo'] = list(hist.xerrl())
    data['x_err_hi'] = list(hist.xerrh())
    data['y'] = list(hist.y())
    data['y_err'] = list(hist.yerravg())
    data['y_err_lo'] = list(hist.yerrl())
    data['y_err_hi'] = list(hist.yerrh())

    table = Table(data)
    return table


def TH2_to_FITS_header(h, flipx=True):
    """Create FITS header assuming TH2 or SkyHist that represents an image
    in Galactic CAR projection with reference point at GLAT = 0,
    as is the case for HESS SkyHists.

    Formulae and variable names taken from Plotters::SkyHistToFITS()
    in $HESSROOT/plotters/src/FITSUtils.C
    """
    # Compute FITS projection header parameters
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    centerbinx = int((nx + 1) / 2)
    centerbiny = int((ny + 1) / 2)
    crval1 = h.GetXaxis().GetBinCenter(centerbinx)
    crval2 = 0
    cdelt1 = (h.GetXaxis().GetXmax() - h.GetXaxis().GetXmin()) / nx
    cdelt2 = (h.GetYaxis().GetXmax() - h.GetYaxis().GetXmin()) / ny
    crpix1 = centerbinx
    crpix2 = centerbiny - h.GetYaxis().GetBinCenter(centerbiny) / cdelt2
    if flipx:
        cdelt1 *= -1
    # Fill dictionary with FITS header keywords
    header = dict()
    header['NAXIS'] = 2
    header['NAXIS1'], header['NAXIS2'] = nx, ny
    header['CTYPE1'], header['CTYPE2'] = 'GLON-CAR', 'GLAT-CAR'
    header['CRVAL1'], header['CRVAL2'] = crval1, crval2
    header['CRPIX1'], header['CRPIX2'] = crpix1, crpix2
    header['CUNIT1'], header['CUNIT2'] = 'deg', 'deg'
    header['CDELT1'], header['CDELT2'] = cdelt1, cdelt2
    return header


def TH2_to_FITS_data(h, flipx=True):
    """Convert TH2 bin values into a numpy array"""
    # @note: Numpy array index order is (y, x), whereas ROOT TH2 has (x, y)
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    # @todo This doesn't work properly:
    # dtype = type(h.GetBinContent(0))
    dtype = 'float32'
    array = np.empty((ny, nx), dtype=dtype)
    for ix in range(nx):
        for iy in range(ny):
            array[iy, ix] = h.GetBinContent(ix, iy)
    if flipx:
        array = array[:, ::-1]
    return array


def TH2_to_FITS(h, flipx=True):
    """Convert ROOT TH2 to kapteyn.maputils.FITSimage,
    which can easily be exported to a FITS file.

    h : Input 2D ROOT histogram

    Usage example:
    >>> from TH2_to_FITS import TH2_to_FITS
    >>> root_th2 = ROOT.TH2F()
    >>> fits_figure = TH2_to_FITS(root_th2)
    >>> fits_figure.writetofits('my_image.fits')
    """
    from kapteyn.maputils import FITSimage
    header = TH2_to_FITS_header(h, flipx)
    if header['CDELT1'] > 0:
        warnings.warn('CDELT1 > 0 might not be handled properly.'
                      'A TH2 representing an astro image should have '
                      'a reversed x-axis, i.e. xlow > xhi')
    data = TH2_to_FITS_data(h, flipx)
    return FITSimage(externaldata=data, externalheader=header)


def tree_to_table(tree):
    """Convert a ROOT TTree to an astropy Table.
    
    Parameters
    ----------
    
    Returns
    -------
    table : astropy.table.Table
        
    """
    from rootpy import asrootpy
    tree = asrootpy(tree)
    array = tree.to_array()

    
    from rootpy.io import open
    from rootpy.root2array import tree_to_recarray
    print('Reading %s' % infile)
    file = open(infile)
    tree_name = 'TableAllMu_WithoutNan' # 'ParTree_Postselect'
    print('Getting %s' % tree_name)
    tree = file.get(tree_name, ignore_unsupported=True)
    print('Converting tree to recarray')
    array = tree_to_recarray(tree)
    print('Converting recarray to atpy.Table')
    table = recarray_to_table(array)

    print('Removing empty columns:')
    empty_columns = []
    for col in table.columns:
        #if table['col'].min() == table['col'].max()
        if (table['col'] == 0).all():
            empty_columns.append(col)
    print(empty_columns)
    table.remove_columns(empty_columns)
    
    for name in array.dtype.names:
        # FITS can't save these types.
        data = array[name]
        if data.dtype == np.dtype('uint64'):
            data = data.copy().astype('int64')
        table.add_column(name, data)
    return table
