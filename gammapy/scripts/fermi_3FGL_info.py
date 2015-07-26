# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from __future__ import print_function, division
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
import os
from astropy.units import Quantity
from astropy.io import fits
from ..utils.scripts import get_parser

__all__ = ['fermi_3FGL_info']


def print_info(source_name):
    """
    TODO: Doccomments
    """
    from gammapy.datasets import fermi

    fermi_3fgl_obj = fermi.Fermi3FGLObject(source_name)

    print(source_name)
    print ("\n")
    print ("RA (J2000) " + str(fermi_3fgl_obj.ra))
    print ("Dec (J2000) " + str(fermi_3fgl_obj.dec))
    print ("l " + str(fermi_3fgl_obj.gal_long))
    print ("b " + str(fermi_3fgl_obj.gal_lat))
    print ("Flux " + str(fermi_3fgl_obj.int_flux) + " +/- " + str(fermi_3fgl_obj.unc_int_flux)
           + " ph /cm2 /MeV /s")
    print ("Detection significance: " + str(fermi_3fgl_obj.signif) + " sigma")

