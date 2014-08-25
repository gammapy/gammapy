# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""SExtractor wrapper (limited functionality, but simple to use)

SExtractor: http://www.astromatic.net/software/sextractor

Other SExtractor Python wrappers (not BSD licensed!):

* http://chimera.googlecode.com/svn/trunk/src/chimera/util/sextractor.py
* https://pypi.python.org/pypi/pysex/
* http://gitorious.org/pysextractor/pysextractor/trees/master/pysextractor
"""
from __future__ import print_function, division
import logging
import subprocess
import tempfile
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.table import Table

__all__ = ['sex']


def sex(image,
        image2='',
        catalog_name=None,
        config_name=None,
        parameters_name=None,
        checkimage_name=None,
        detect_thresh=5,
        detect_minarea=1,
        deblend_mincont=1,
        ):
    """Run SExtractor to detect sources in an image.

    Parameters
    ----------
    image : str
        Detection image filename
    image2 : str
        Measurement image filename (same as image if '')
    catalog_name : str
        Output catalog filename
    config_name : str
        Config filename
    parameters_name : str
        Name of the file describing the catalog output parameters
    checkimage_name : str
        Filename for the check-image (TODO: none for '')
    detect_thresh : float
        Detection threshold
    detect_minarea : int
        Minimum number of pixels above threshold
    deblend_mincont : float in range 0 to 1
        Minimum contrast parameter for deblending.

        * 0 = each peak is a single source
        * 1 = no deblending, one source per segment

    Returns
    -------
    catalog : `~astropy.table.Table`
        Catalog of detected objects

    checkimage : `~astropy.io.fits.PrimaryHDU`
        Segmented image

    Examples
    --------
    TODO: look what other Python sextractor wrappers do:

    TODO: where to run the command and put the output files?

    TODO: return filenames or dict with results?
    """
    if catalog_name == None:
        catalog_name = tempfile.mktemp('.fits')

    if checkimage_name == None:
        checkimage_name = tempfile.mktemp('.fits')

    if config_name == None:
        config_name = get_pkg_data_filename('sex.cfg')

    if parameters_name == None:
        parameters_name = get_pkg_data_filename('sex.param')

    logging.info('Running SExtractor')
    logging.info('INPUT  image: {0}'.format(image))
    logging.info('INPUT  image2: {0}'.format(image2))
    logging.info('INPUT  config_name: {0}'.format(config_name))
    logging.info('INPUT  parameters_name: {0}'.format(parameters_name))
    logging.info('OUTPUT catalog_name: {0}'.format(catalog_name))
    logging.info('OUTPUT checkimage_name: {0}'.format(checkimage_name))

    cmd = ['sex', image, image2,
           '-c', config_name,
           '-catalog_name', catalog_name,
           '-parameters_name', parameters_name,
           '-checkimage_name', checkimage_name,
           '-detect_thresh', str(detect_thresh),
           '-detect_minarea', str(detect_minarea),
           '-deblend_mincont', str(deblend_mincont)
           ]
    logging.info('Executing the following command now:\n\n{0}\n'.format(' '.join(cmd)))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Read output files
    catalog = Table.read(catalog_name)
    checkimage = fits.open(checkimage_name)[0]
    logging.info('Number of objects detected: {0}'.format(len(catalog)))

    return catalog, checkimage
