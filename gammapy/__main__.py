# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from gammapy.scripts.info import print_info

# create the top-level parser
parser = argparse.ArgumentParser(prog='gammapy')
parser.set_defaults(func=lambda x: parser.print_help())
subparsers = parser.add_subparsers(title='Available Gammapy commands',
                                   dest='subparser_name')

# create the parser for the "info" command
parser_info = subparsers.add_parser('info', help='Print info about Gammapy.')
parser_info.add_argument('--version', action='store_true',
                    help='Display Gammapy version number')
parser_info.add_argument('--dependencies', action='store_true',
                    help='Dsiplay available dependencies and versions')
parser_info.add_argument('--all', action='store_true',
                    help='Display all info')
parser_info.set_defaults(func=print_info)

args = parser.parse_args()
args.func(args)