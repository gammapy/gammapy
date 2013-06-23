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
import warnings
import numpy as np


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
