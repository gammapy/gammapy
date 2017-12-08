# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse

# This trick is taken from the conda command line tool, which does a delayed
# import on call of the subcommand
def call_command(args):
    relative_mod, func_name = args.func.rsplit('.', 1)
    from importlib import import_module
    module = import_module(relative_mod, 'gammapy')
    exit_code = getattr(module, func_name)(args)
    return exit_code


def main():
    # create the top-level parser
    parser = argparse.ArgumentParser(
        prog='gammapy',
        description='Gammapy is a tool for high level analysis of astronomical'
                    ' gamma-ray data.',
        )

    parser.set_defaults(func=lambda x: parser.print_help())
    sub_parsers = parser.add_subparsers(
        title='Available Gammapy commands',
        dest='command',
        help='Use gammapy <command> --help to learn more.'
        )

    sub_parsers.required = True

    configure_parse_info(sub_parsers)
    configure_check_info(sub_parsers)

    args = parser.parse_args()
    call_command(args)


def configure_parse_info(sub_parsers):
    # description is shown when gammapy <command> --help is called
    description = """
    Display information about current Gammapy install.
    """

    # help is the short description shown when gammapy --help is called
    help = 'Display information about current Gammapy install.'

    p = sub_parsers.add_parser(
        name='info',
        description=description,
        help=help,
        )

    p.add_argument('--version', action='store_true',
                    help='Print Gammapy version number')
    p.add_argument('--dependencies', action='store_true',
                    help='Print available versions of dependencies')
    p.add_argument('--all', action='store_true',
                    help='Display all info')

    # the string is the entry point for the subcommand function
    p.set_defaults(func='.scripts.info.main')


def configure_check_info(sub_parsers):
    p = sub_parsers.add_parser(
        name='check',
        description='Check current Gammapy install.',
        help='Check current Gammapy install.',
        )

    p.set_defaults(func='.scripts.check.main')


if __name__ == '__main__':
    main()