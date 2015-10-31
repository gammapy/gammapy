# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ..utils.scripts import get_parser, set_up_logging_from_args

__all__ = ['image_pfmap']


def image_pfmap_main(args=None):
    parser = get_parser(image_pfmap)
    parser.add_argument('infile', type=str,
                        help='Input file. Either an individual FITS file or a batch file.')
    parser.add_argument('-s', '--skymap_size', type=float, default=6.,
                        help='Diameter of the sky map in degrees')
    parser.add_argument('-b', '--bin_size', type=float, default=0.03,
                        help='Bin size in degrees')
    parser.add_argument('-p', '--analysis_position', type=str, default=None,
                        help='Center of the skymap in RA and Dec (J2000) in degrees. '
                             'Format: \'(RA, Dec)\', including the quotation marks. '
                             'If no center is given, the source position from the first input file is used.')
    parser.add_argument('-r', '--oversampling_radius', type=float, default=0.125,
                        help='Radius used to correlated the sky maps in degrees.')
    parser.add_argument('--ring_bg_radii', type=str, default='(.3, .7)',
                        help='Inner and outer radius of the ring used for the ring background. '
                             'Format \'(r_in, r_out)\', including the quotation marks.')
    parser.add_argument('-w', '--write_output', action='store_true', default=False,
                        help='Write results to FITS files in current directory.')
    parser.add_argument('--acceptance_correction', action='store_true', default=False,
                        help='Do not correct skymaps for FoV acceptance.')
    parser.add_argument('-t', '--template_background', type=str, default=None,
                        help='Bankfile with template background eventlists.')
    parser.add_argument('--graphical_output', action='store_true', default=False,
                        help='Switch off graphical output.')
    parser.add_argument("-l", "--loglevel", default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help="Set the logging level")
    args = parser.parse_args(args)

    set_up_logging_from_args(args)

    image_pfmap(**vars(args))


def image_pfmap(infile,
                skymap_size,
                bin_size,
                analysis_position,
                oversampling_radius,
                ring_bg_radii,
                write_output,
                acceptance_correction,
                template_background,
                graphical_output,
                loglevel):
    """Create sky maps from VHE event lists.

    TODO: document
    """
    from ..utils.pyfact import create_sky_map

    create_sky_map(input_file_name=infile,
                   skymap_size=skymap_size,
                   skymap_bin_size=bin_size,
                   r_overs=oversampling_radius,
                   ring_bg_radii=ring_bg_radii,
                   template_background=template_background,
                   skymap_center=analysis_position,
                   write_output=write_output,
                   fov_acceptance=acceptance_correction,
                   do_graphical_output=graphical_output,
                   loglevel=loglevel)
