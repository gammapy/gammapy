# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions to convert ROOT data to numpy / FITS data.
"""
from __future__ import print_function, division
import warnings
import numpy as np
from astropy.utils.compat.odict import OrderedDict
from astropy.io import fits
from astropy.table import Table

__all__ = ['hist1d_to_table',
           'graph1d_to_table',
           'TH2_to_FITS_header',
           'TH2_to_FITS_data',
           'TH2_to_FITS',
           ]

__doctest_skip__ = ['TH2_to_FITS']


def hist1d_to_table(hist):
    """Convert 1D ROOT histogram into astropy table.

    Parameters
    ----------
    hist : ROOT.TH1
        ROOT histogram

    Returns
    -------
    table : `~astropy.table.Table`
        Astropy table
    """
    bins = range(1, hist.GetNbinsX() + 1)

    data = OrderedDict()

    names = [('x', 'GetBinCenter'),
             ('x_bin_lo', 'GetBinLowEdge'),
             ('x_bin_width', 'GetBinWidth'),
             ('y', 'GetBinContent'),
             ('y_err', 'GetBinError'),
             ('y_err_lo', 'GetBinErrorLow'),
             ('y_err_hi', 'GetBinErrorUp'),
             ]

    for column, method in names:
        try:
            getter = getattr(hist, method)
            data[column] = [getter(i) for i in bins]
        # Note: `GetBinErrorLow` is not available in old ROOT versions!?
        except AttributeError:
            pass

    table = Table(data)
    table['x_bin_hi'] = table['x_bin_lo'] + table['x_bin_width']

    return table


def graph1d_to_table(graph):
    """Convert ROOT TGraph to an astropy Table.

    Parameters
    ----------
    graph : ROOT.TGraph
        ROOT graph

    Returns
    -------
    table : `~astropy.table.Table`
        Astropy table
    """
    bins = range(0, graph.GetN())

    data = OrderedDict()

    names = [('x', 'GetX'),
             ('x_err', 'GetEX'),
             ('x_err_lo', 'GetEXlow'),
             ('x_err_hi', 'GetEXhigh'),
             ('y', 'GetY'),
             ('y_err', 'GetEY'),
             ('y_err_lo', 'GetEYlow'),
             ('y_err_hi', 'GetEYhigh'),
             ]

    for column, method in names:
        try:
            buffer_ = getattr(graph, method)()
            data[column] = [buffer_[i] for i in bins]
        except IndexError:
            pass

    table = Table(data)
    return table


def TH2_to_FITS_header(hist, flipx=True):
    """Create FITS header for a given ROOT histogram.

    Assuming TH2 or SkyHist that represents an image
    in Galactic CAR projection with reference point at GLAT = 0,
    as is the case for HESS SkyHists.

    Formulae and variable names taken from ``Plotters::SkyHistToFITS()``
    in ``$HESSROOT/plotters/src/FITSUtils.C``

    Parameters
    ----------
    hist : ROOT.TH2
        ROOT histogram
    flipx : bool
        Flip x-axis?

    Returns
    -------
    header : `~astropy.io.fits.Header`
        FITS header
    """
    # Compute FITS projection header parameters
    nx, ny = hist.GetNbinsX(), hist.GetNbinsY()
    centerbinx = int((nx + 1) / 2)
    centerbiny = int((ny + 1) / 2)
    crval1 = hist.GetXaxis().GetBinCenter(centerbinx)
    crval2 = 0
    cdelt1 = (hist.GetXaxis().GetXmax() - hist.GetXaxis().GetXmin()) / nx
    cdelt2 = (hist.GetYaxis().GetXmax() - hist.GetYaxis().GetXmin()) / ny
    crpix1 = centerbinx
    crpix2 = centerbiny - hist.GetYaxis().GetBinCenter(centerbiny) / cdelt2
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


def TH2_to_FITS_data(hist, flipx=True):
    """Convert TH2 bin values into a numpy array.

    Parameters
    ----------
    hist : ROOT.TH2
        ROOT histogram

    Returns
    -------
    data : array
        Histogram data
    """
    # @note: Numpy array index order is (y, x), whereas ROOT TH2 has (x, y)
    nx, ny = hist.GetNbinsX(), hist.GetNbinsY()
    # @todo This doesn't work properly:
    # dtype = type(hist.GetBinContent(0))
    dtype = 'float32'
    array = np.empty((ny, nx), dtype=dtype)
    for ix in range(nx):
        for iy in range(ny):
            array[iy, ix] = hist.GetBinContent(ix, iy)
    if flipx:
        array = array[:, ::-1]

    return array


def TH2_to_FITS(hist, flipx=True):
    """Convert ROOT 2D histogram to FITS format.

    Parameters
    ----------
    hist : ROOT.TH2
        2-dim ROOT histogram

    Returns
    -------
    hdu : `~astropy.io.fits.ImageHDU`
        Histogram in FITS format.

    Examples
    --------
    >>> import ROOT
    >>> from gammapy.utils.root import TH2_to_FITS
    >>> root_hist = ROOT.TH2F()
    >>> fits_hdu = TH2_to_FITS(root_hist)
    >>> fits_hdu.writetofits('my_image.fits')
    """
    header = TH2_to_FITS_header(hist, flipx)
    if header['CDELT1'] > 0:
        warnings.warn('CDELT1 > 0 might not be handled properly.'
                      'A TH2 representing an astro image should have '
                      'a reversed x-axis, i.e. xlow > xhi')
    data = TH2_to_FITS_data(hist, flipx)
    hdu = fits.ImageHDU(data=data, header=header)
    return hdu


def tree_to_table(tree, tree_name):
    """Convert a ROOT TTree to an astropy Table.

    Parameters
    ----------
    tree : ROOT.TTree
        ROOT TTree

    Returns
    -------
    table : `~astropy.table.Table`
        ROOT tree data as an astropy table.
    """
    from rootpy import asrootpy
    from rootpy.io import open
    from rootpy.root2array import tree_to_recarray

    tree = asrootpy(tree)
    array = tree.to_array()

    file = open(infile)
    tree_name = 'TableAllMu_WithoutNan'  # 'ParTree_Postselect'
    tree = file.get(tree_name, ignore_unsupported=True)
    array = tree_to_recarray(tree)
    table = recarray_to_table(array)

    # Remove empty columns
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
