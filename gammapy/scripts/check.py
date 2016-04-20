# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Command line tool to check various things about a Gammapy installation.

This file is called `check` instead of `test` to prevent confusion
for developers and the test runner from including it in test collection.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import warnings
from ..utils.scripts import get_parser, set_up_logging_from_args

__all__ = [
    'run_tests',
    'run_log_examples',
]

log = logging.getLogger(__name__)


def check_main(args=None):
    parser = get_parser(run_tests)
    parser.description = 'Check various things about the Gammapy installation'
    subparsers = parser.add_subparsers(help='commands', dest='subparser_name')

    test_parser = subparsers.add_parser('runtests', help='Run tests')
    test_parser.add_argument('--package', type=str, default=None,
                             help='Package to test')

    log_parser = subparsers.add_parser('logging', help='Print logging examples (for debugging)')
    log_parser.add_argument("-l", "--loglevel", default='info',
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            help="Set the logging level")

    data_parser = subparsers.add_parser('fitsexport', help='Test fits data')
    data_parser.add_argument('--directory', default='out', type=str,
                            help="FITS export output folder")


    args = parser.parse_args(args)
    set_up_logging_from_args(args)

    if args.subparser_name == 'runtests':
        del args.subparser_name
        run_tests(**vars(args))
    elif args.subparser_name == 'logging':
        del args.subparser_name
        run_log_examples(**vars(args))
    elif args.subparser_name == 'fitsexport':
        del args.subparser_name
        run_test_fitsexport(**vars(args))
    else:
        parser.print_help()
        exit(0)


def run_tests(package):
    """Run Gammapy tests."""
    import gammapy
    gammapy.test(package, verbose=True)


def run_log_examples():
    """Run some example code that generates log output.

    This is mainly useful for debugging logging output from Gammapy.
    """
    log.debug('this is log.debug() output')
    log.info('this is log.info() output')
    log.warning('this is log.warning() output')
    warnings.warn('this is warnings.warn() output')


def run_test_fitsexport(directory):
    """Run example analysis to test a fits data production

    hap-data-fits-export crab has to be run in order to produce the example data
    """
    log.info('Running test analysis of fits data')
    from gammapy.data import DataStore
    from gammapy.datasets import gammapy_extra
    from gammapy.utils.scripts import read_yaml
    from gammapy.spectrum.spectrum_pipe import run_spectrum_analysis_using_config
    from gammapy.spectrum.results import SpectrumResult

    s = DataStore.from_dir(directory)
    print(s.info())
    configfile = gammapy_extra.filename(
        'test_datasets/spectrum/spectrum_analysis_example.yaml')
    config = read_yaml(configfile)
    config['extraction']['data']['datastore'] = directory
    config['extraction']['data']['runlist'] = [23523, 23526, 23559, 23592]

    fit, analysis = run_spectrum_analysis_using_config(config)
    res = SpectrumResult(fit=fit.result, stats=analysis.observations.total_spectrum.spectrum_stats)
    print(res.to_table())

