# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
TODO: clean up this old script.

A script to check if CameraX/YEvent are the gnomonic projection of Alt/Az.

This script adds four columns to the .csv result file of the partial_ttree_to_csv.C script :
GnomAlt & GnomAz that are the reverse gnomonic projection of CameraX/YEvent (the center of
the projective plane is set to the pointing position) and NomX & NomY that are the gnomonic
projection of Alt/Az.

This script takes a input a csv table containing the CameraXEvent, CameraYEvent, AzEvent, AltEvent,
AzSystem, AltSystem. Typically an output of the tools/partial_tree_to_csv.C script
"""
import sys
import logging
import numpy as np


def tan_world_to_pix(lon, lat, lon_center, lat_center):
    """Hand-coded TAN (a.k.a. tangential or Gnomonic) projection.

    TODO: test against these implementations:
    - http://docs.astropy.org/en/latest/_generated/astropy.modeling.projections.Sky2Pix_TAN.html
    - http://www.astro.rug.nl/software/kapteyn/allsky.html#fig-5-gnomonic-projection-tan

    Parameters
    ----------
    lon, lat : floats or array-like
        Event coordinates
    lon_center, lat_center: floats or array-like
        System center coordinates

    Returns
    -------
    x, y : arrays
        Nominal coordinates
    """
    alpha = np.radians(lon)
    delta = np.radians(lat)
    alpha_center = np.radians(lon_center)
    delta_center = np.radians(lat_center)

    A = np.cos(delta) * np.cos(alpha - alpha_center)
    F = 1. / (np.sin(delta_center) * np.sin(delta) + A * np.cos(delta_center))

    x = F * (np.cos(delta_center) * np.sin(delta) - A * np.sin(delta_center))
    y = F * np.cos(delta) * np.sin(alpha - alpha_center)

    return x, y


def add_tan_pix_coordinates(in_file_name, out_file_name):
    """Compute x, y nominal coordinates from alt, az
    and add as columns to a CSV file"""
    logging.info('Reading file: {0}'.format(in_file_name))
    data = np.recfromcsv(in_file_name)
    az = data['azevent']
    alt = data['altevent']
    alts = data['altsystem']
    azs = data['azsystem']
    infile = file(in_file_name)

    logging.info('Writing file: {0}'.format(out_file_name))
    outfile = file(out_file_name, 'w')
    names = infile.readline().split()
    names.append(',Nomx,Nomy\n')
    line = ' '.join(names)
    line = line.replace(' ', '')
    outfile.write(line)

    for ii in np.arange(0, len(alts) - 1, 1):
        noms = tan_world_to_pix(az[ii], alt[ii], azs[ii], alts[ii])
        values = infile.readline().split()
        values.append(',%s,%s\n' % (str(noms[0]), str(noms[1])))
        line = ' '.join(values)
        line = line.replace(' ', '')

        outfile.write(line)

    infile.close()
    outfile.close()


def add_tan_world_coordinates(csv_file, outfile):
    """Compute alt, az from x, y nominal coordinates
    and add as columns to a CSV file."""
    from kapteyn.wcs import Projection

    logging.info('Reading file: {0}'.format(csv_file))
    data = np.recfromcsv(csv_file)
    camX = data['cameraxevent']
    camY = data['camerayevent']
    alts = data['altsystem']
    azs = data['azsystem']
    infile = file(csv_file)

    logging.info('Writing file: {0}'.format(outfile))
    outfile = file(outfile, 'w')
    names = infile.readline().split()
    names.append(',GnomAz,GnomAlt\n')
    line = ' '.join(names)
    line = line.replace(' ', '')
    outfile.write(line)

    for ii in np.arange(0, len(alts) - 1, 1):
        header = {'NAXIS':  2,
                  'NAXIS1':  100, 'NAXIS2': 100,
                  'CTYPE1': 'RA---TAN',
                  'CRVAL1':  azs[ii], 'CRPIX1': 0, 'CUNIT1': 'deg',
                  'CDELT1':  np.degrees(1), 'CTYPE2': 'DEC--TAN',
                  'CRVAL2':  alts[ii], 'CRPIX2': 0,
                  'CUNIT2': 'deg', 'CDELT2': np.degrees(1),
                  }
        projection = Projection(header)
        gnoms = projection.toworld((camY[ii], camX[ii]))

        values = infile.readline().split()
        values.append(',%s,%s\n' % (str(gnoms[0]), str(gnoms[1])))
        line = ' '.join(values)
        line = line.replace(' ', '')

        outfile.write(line)

    infile.close()
    outfile.close()


if __name__ == '__main__':
    usage = 'nom_to_altaz.py infile outfile'
    if len(sys.argv) != 3:
        print(usage)
        exit()

    csv_file = sys.argv[1]
    outfile = sys.argv[2]
    add_tan_pix_coordinates(csv_file, 'gnomonic_test.csv')
    add_tan_world_coordinates('gnomonic_test.csv', outfile)
