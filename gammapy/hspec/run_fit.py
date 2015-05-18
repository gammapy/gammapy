# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to run Sherpa spectrum fit.
"""
from ..utils.scripts import get_parser


def main(args=None):
    parser = get_parser(sherpa_spec_fit)
    parser.add_argument('in_files', type=str, nargs='+',
                        help='Input pha files, typically run?????.pha, with the background, '
                             'RMF and ARF defined in their header (any wildcard accepted)')
    parser.add_argument('model', type=str, nargs='+',
                        help='Sherpa model(s) to be fitted. Pre-setup models: PowLaw1D, logparabola, plexpcutoff; '
                             'or multiple components. Example: PowLaw1D+logparabola')
    parser.add_argument('--noplot', action='store_true',
                        help='Do not produce the spectral plot (no graphical output).')
    parser.add_argument('--manual', action='store_true',
                        help='Set initial values and freeze parameters manually.'
                             'This is default in case a non pre-setup model is called.')
    parser.add_argument('--reproj', type=int, default=3,
                        help='Reprojection level to use 1 to 3, default is 3. level 0 is no reprojection at all')
    parser.add_argument('--do-conf', action='store_true',
                        help='Compute confidence values for the fitted parameters')
    args = parser.parse_args(args)
    sherpa_spec_fit(**vars(args))


def sherpa_spec_fit(in_files,
                    model,
                    noplot,
                    manual,
                    reproj,
                    do_conf):
    """Reads a set of pha files and performs a maximum likelihood fit.
    """
    import logging

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    # from sherpa.astro.ui import *  # TEST if needed (only when outside sherpa env)
    import sherpa.astro.ui as sau
    from . import load_model
    from . import make_plot
    from .specsource import SpecSource

    logger = logging.getLogger("sherpa")
    logger.setLevel(logging.ERROR)

    # Read and load the data and model:
    list_data = []

    emax = 9e10  # Maximum energy taken into account for the fit --> Compute properly (as a function of the max E event?)
    # We don't use a specific maximum reduced statistic value since we don't expect the cstat to be anywhere near the large number limit
    sau.set_conf_opt("max_rstat", 10000)

    p1 = load_model.load_model(model[0])  # [-1] #load model returns an array of model components
    # where the last component is the total model
    spec = SpecSource('SRC', in_files)

    if reproj == 0:
        myspec = spec
    elif reproj == 3:
        myspec = spec.reproject(nbins={'offset': 5, 'eff': 10, 'zen': 10})
    elif reproj == 2:
        myspec = spec.reproject(nbins={'offset': 5, 'eff': 20, 'zen': 12})
    elif reproj == 1:
        myspec = spec.reproject(nbins={'offset': 25, 'eff': 40, 'zen': 30})
    elif reproj == 4:
        myspec = spec.reproject(nbins={'offset': 1, 'eff': 1, 'zen': 1})

    myspec.set_source(p1)

    if manual:
        load_model.set_manual_model(sau.get_model(datid))  # Parameters for all runs are linked
    else:
        print('Using default initial values for model parameters')

    myspec.fit(do_conf=do_conf)

    if noplot:
        quit()
    make_plot.make_plot(list_data, p1)

    raw_input('Press <ENTER> to continue')
