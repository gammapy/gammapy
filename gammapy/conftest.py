# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.pytest_plugins import *

## Uncomment the following line to treat all DeprecationWarnings as
## exceptions
# enable_deprecations_as_exceptions()

## Uncomment and customize the following lines to add/remove entries
## from the list of packages for which version numbers are displayed
## when running the tests ... this was added in Astropy 1.0
try:
    del PYTEST_HEADER_MODULES['h5py']

    PYTEST_HEADER_MODULES['cython'] = 'cython'
    PYTEST_HEADER_MODULES['pandas'] = 'pandas'
    PYTEST_HEADER_MODULES['skimage'] = 'skimage'
    PYTEST_HEADER_MODULES['sklearn'] = 'sklearn'
    PYTEST_HEADER_MODULES['uncertainties'] = 'uncertainties'

    PYTEST_HEADER_MODULES['astropy'] = 'astropy'
    PYTEST_HEADER_MODULES['gammapy'] = 'gammapy'
    PYTEST_HEADER_MODULES['gammafit'] = 'gammafit'
    PYTEST_HEADER_MODULES['imageutils'] = 'imageutils'
    PYTEST_HEADER_MODULES['reproject'] = 'reproject'
    PYTEST_HEADER_MODULES['photutils'] = 'photutils'
    PYTEST_HEADER_MODULES['wcsaxes'] = 'wcsaxes'
    PYTEST_HEADER_MODULES['aplpy'] = 'aplpy'
    PYTEST_HEADER_MODULES['pyregion'] = 'pyregion'

    # `ginga` doesn't have a __version__ attribute yet, so this won't work:
    #PYTEST_HEADER_MODULES['ginga'] = 'ginga'

except NameError:  # astropy < 1.0
    pass
