# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from ..utils.scripts import get_parser, set_up_logging_from_args

__all__ = ['spectrum_pfsim']

log = logging.getLogger(__name__)


def spectrum_pfsim_main(args=None):
    parser = get_parser(spectrum_pfsim)
    parser.add_argument('arf', type=str,
                        help='Input ARF file.')
    parser.add_argument('-t', '--exposure_time', type=float, default=.5,
                        help='Exposure time in hours')
    parser.add_argument('-f', '--flux', type=float, default=.1,
                        help='Flux in units of Crab')
    parser.add_argument('-r', '--rmf_file', type=str, default=None,
                        help='Response matrix file (RMF), optional')
    parser.add_argument('-e', '--extra_file', type=str, default=None,
                        help='Extra file with auxiliary information e.g. bg-rate, psf, etc.')
    parser.add_argument('-o', '--output_filename_base', type=str, default=None,
                        help='Output filename base. If set, output files will be written')
    parser.add_argument('--write_pha', action='store_true', default=False,
                        help='Write photon PHA file')
    parser.add_argument('--graphical_output', action='store_true', default=False,
                        help='Switch off graphical output')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)

    set_up_logging_from_args(args)

    spectrum_pfsim(**vars(args))


def spectrum_pfsim(arf,
                   exposure_time,
                   flux,
                   rmf_file,
                   extra_file,
                   output_filename_base,
                   write_pha,
                   graphical_output,
                   loglevel):
    """Simulates IACT eventlist using an ARF file.

    TODO: document
    """
    from ..utils.pyfact import sim_evlist

    sim_evlist(flux=flux,
               obstime=exposure_time,
               arf=arf,
               rmf=rmf_file,
               extra=extra_file,
               output_filename_base=output_filename_base,
               write_pha=write_pha,
               do_graphical_output=graphical_output,
               loglevel=loglevel)
