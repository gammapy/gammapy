# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from .hpx_utils import HPX


def read_energy_bounds(hdu):
    """ Reads and returns the energy bin edges from a FITs HDU
    """
    nebins = len(hdu.data)
    ebin_edges = np.ndarray((nebins + 1))
    ebin_edges[0:-1] = np.log10(hdu.data.field("E_MIN")) - 3.
    ebin_edges[-1] = np.log10(hdu.data.field("E_MAX")[-1]) - 3.
    return ebin_edges


def read_spectral_data(hdu):
    """ Reads and returns the energy bin edges, fluxes and npreds from
    a FITs HDU
    """
    ebins = read_energy_bounds(hdu)
    fluxes = np.ndarray((len(ebins)))
    try:
        fluxes[0:-1] = hdu.data.field("E_MIN_FL")
        fluxes[-1] = hdu.data.field("E_MAX_FL")[-1]
        npreds = hdu.data.field("NPRED")
    except:
        fluxes = np.ones((len(ebins)))
        npreds = np.ones((len(ebins)))
    return ebins, fluxes, npreds


def write_maps(primary_map, maps, outfile):
    hdu_images = [primary_map.create_primary_hdu()]
    for k, v in sorted(maps.items()):
        hdu_images += [v.create_image_hdu(k)]

    hdulist = pyfits.HDUList(hdu_images)
    for h in hdulist:
        from gammapy import __version__
        h.header['CREATOR'] = 'gammapy ' + __version__
    hdulist.writeto(outfile, clobber=True)


def write_fits_image(data, wcs, outfile):
    hdu_image = pyfits.PrimaryHDU(data, header=wcs.to_header())
    hdulist = pyfits.HDUList([hdu_image])
    hdulist.writeto(outfile, clobber=True)


def write_hpx_image(data, hpx, outfile, extname="SKYMAP"):
    hpx.write_fits(data, outfile, extname, clobber=True)


def read_projection_from_fits(fitsfile, extname=None):
    """
    Load a WCS or HPX projection.
    """
    f = pyfits.open(fitsfile)
    nhdu = len(f)
    # Try and get the energy bounds
    try:
        ebins = read_energy_bounds(f['EBOUNDS'])
    except:
        ebins = None

    if extname is None:
        # If there is an image in the Primary HDU we can return a WCS-based
        # projection
        if f[0].header['NAXIS'] != 0:
            proj = pywcs.WCS(f[0].header)
            return proj, f, f[0]
    else:
        if f[extname].header['XTENSION'] == 'IMAGE':
            proj = pywcs.WCS(f[extname].header)
            return proj, f, f[extname]
        elif f[extname].header['XTENSION'] == 'BINTABLE':
            try:
                if f[extname].header['PIXTYPE'] == 'HEALPIX':
                    proj = HPX.create_from_header(f[extname].header, ebins)
                    return proj, f, f[extname]
            except:
                pass
        return None, f, None

    # Loop on HDU and look for either an image or a table with HEALPix data
    for i in range(1, nhdu):
        # if there is an image we can return a WCS-based projection
        if f[i].header['XTENSION'] == 'IMAGE':
            proj = pywcs.WCS(f[i].header)
            return proj, f, f[i]
        elif f[i].header['XTENSION'] == 'BINTABLE':
            try:
                if f[i].header['PIXTYPE'] == 'HEALPIX':
                    proj = HPX.create_from_header(f[i].header, ebins)
                    return proj, f, f[i]
            except:
                pass

    return None, f, None


def write_tables_to_fits(filepath, tablelist, clobber=False,
                         namelist=None, cardslist=None, hdu_list=None):
    """
    Write some astropy.table.Table objects to a single fits file
    """
    outhdulist = [pyfits.PrimaryHDU()]
    rmlist = []
    for i, table in enumerate(tablelist):
        ft_name = "%s._%i" % (filepath, i)
        rmlist.append(ft_name)
        try:
            os.unlink(ft_name)
        except:
            pass
        table.write(ft_name, format="fits")
        ft_in = pyfits.open(ft_name)
        if namelist:
            ft_in[1].name = namelist[i]
        if cardslist:
            for k, v in cardslist[i].items():
                ft_in[1].header[k] = v
        ft_in[1].update()
        outhdulist += [ft_in[1]]

    if hdu_list is not None:
        for h in hdu_list:
            outhdulist.append(h)

    pyfits.HDUList(outhdulist).writeto(filepath, clobber=clobber)
    for rm in rmlist:
        os.unlink(rm)
