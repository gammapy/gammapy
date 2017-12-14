# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

from ..import version
from ..utils.scripts import set_up_logging_from_args


# This trick is taken from the conda command line tool, which does a delayed
# import on call of the subcommand
def do_call(args, parser):
    relative_mod, func_name = args.func.rsplit('.', 1)
    from importlib import import_module
    module = import_module(relative_mod, 'gammapy')
    exit_code = getattr(module, func_name)(args, parser)
    return exit_code


def cmd_main(args, parser):
    """Command executed when gammapy tool is called without subcommand"""
    if args.version:
        print('gammapy {}'.format(version.version))
    else:
        parser.print_help()
    return 0


def generate_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(
        prog='gammapy',
        description='Gammapy is a toolbox for high level analysis of astronomical'
                    ' gamma-ray data.',
    )

    parser.add_argument(
        '--loglevel',
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help="Set the logging level",
    )

    parser.add_argument(
        '--version',
        help='Show the gammapy version number and exit',
        action='store_true',
    )

    parser.set_defaults(func='.scripts.main.cmd_main')

    sub_parsers = parser.add_subparsers(
        title='Available gammapy commands',
        dest='command',
        help='Use gammapy <command> --help to learn more.'
    )

    configure_parse_info(sub_parsers)
    configure_parse_check(sub_parsers)
    return parser


def main(*args):
    parser = generate_parser()
    args = parser.parse_args()
    set_up_logging_from_args(args)
    exit_code = do_call(args, parser)
    return exit_code


def configure_parse_info(sub_parsers):
    # description is shown when gammapy <command> --help is called
    description = """
    Display information about current gammapy install and environment.
    """

    # help is the short description shown when gammapy --help is called
    help = 'Display information about current gammapy install and environment.'

    p = sub_parsers.add_parser(
        name='info',
        description=description,
        help=help,
    )

    p.add_argument(
        '--system',
        action='store_true',
        help='List gammapy relevant environment variables'
    )

    p.add_argument(
        '--dependencies',
        action='store_true',
        help='Show available versions of dependencies'
    )

    p.add_argument(
        '--version',
        action='store_true',
        help='Show detailed gammapy version info'
    )

    p.add_argument(
        '--all',
        action='store_true',
        help='Display all info'
    )

    # the string is the entry point for the subcommand function
    p.set_defaults(func='.scripts.info.cmd_info')


def configure_parse_check(sub_parsers):
    p = sub_parsers.add_parser(
        name='check',
        description='Check current gammapy install.',
        help='Check current gammapy install.',
    )

    sub_parsers_check = p.add_subparsers(
        title='Available check commands',
        help='Use gammapy check <command> --help to learn more.'
    )

    p_test = sub_parsers_check.add_parser('runtests', help='Run Gammapy tests')
    p_test.add_argument('--package', type=str, default=None,
                        help='Subpackage to test')
    p_test.set_defaults(func='.scripts.check.cmd_tests')

    p_log = sub_parsers_check.add_parser(
        name='logging',
        help='Print logging examples (for debugging)')

    p_log.set_defaults(func='.scripts.check.cmd_log_examples')
