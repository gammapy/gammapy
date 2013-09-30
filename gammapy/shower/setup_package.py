import os.path
from os.path import join
from distutils.core import Extension
from astropy import setup_helpers

SHOWER_ROOT = os.path.relpath(os.path.dirname(__file__))

def get_package_data():
    return {'gammapy.shower': ['include/*.h']}

def get_extensions():
    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg['sources'] = [join(SHOWER_ROOT, 'src/fits_shower_images.c')]
    cfg['include_dirs'] = [join(SHOWER_ROOT, 'include')]
    cfg['libraries'] = ['cfitsio']
    return [Extension(str('./gammapy/shower/_fits_shower_images'), **cfg)]
