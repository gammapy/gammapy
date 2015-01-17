# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ..utils.scripts import get_parser

__all__ = ['pfspec']


def main(args=None):
    parser = get_parser(pfspec)
    # TODO: fix next argument
    parser.add_argument('input_file_names', type=str,
                        help='Input file names.')
    parser.add_argument('-p', '--analysis_position', type=str, default=None,
                        help='Center of the skymap in RA and Dec (J2000) in degrees. '
                        'Format: \'(RA, Dec)\', including the quotation marks. '
                        'If no center is given, the source position from the first input file is used.')
    parser.add_argument('-r', '--analysis_radius', type=float, default=0.125,
                        help='Aperture for the analysis in degrees.')
    parser.add_argument('-m', '--match_rmf', type=str, default=None,
                        help='RMF filename to which the average PHA file binning is matched.')
    parser.add_argument('-d', '--data_dir', type=str, default='',
                        help='Directory in which the data is located. '
                             'Will be added as prefix to the entries in the bankfile.')
    parser.add_argument('-w', '--write_output', action='store_true', default=False,
                        help='Write results to FITS files in current directory.')
    parser.add_argument('--graphical_output', action='store_true', default=False,
                        help='Switch off graphical output.')
    parser.add_argument('-l', '--loglevel', type=str, default='INFO',
                        help='Amount of logging e.g. DEBUG, INFO, WARNING, ERROR.')
    args = parser.parse_args(args)
    pfspec(**vars(args))


def pfspec(input_file_names,
           analysis_position,
           analysis_radius,
           match_rmf,
           data_dir,
           write_output,
           graphical_output,
           loglevel):
    """Create spectra from VHE event lists in FITS format.

    prog [options] FILE [ARF RMF]
    FILE can either be an indiviual .fits/.fits.gz file or a batch file.
    In case it is a individual file, the ARF and RMF must also be specified.
    The bankfile must contain three columns: data file, ARF, and RMF.

    TODO: document
    """
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    from ..utils.pyfact import create_spectrum

    create_spectrum(input_file_names=input_file_names,
                    analysis_position=analysis_position,
                    analysis_radius=analysis_radius,
                    match_rmf=match_rmf,
                    datadir=data_dir,
                    write_output_files=write_output,
                    do_graphical_output=graphical_output,
                    loglevel=loglevel)
