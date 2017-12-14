# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from shlex import split

from .cli import generate_parser, do_call

log = logging.getLogger(__name__)


def run_command(command, *arguments):
    """Run a Gammapy command with a given set of command-line interface arguments.

    Parameters
    ----------
    command: str
        Gammapy command or sub-command.
    *arguments: list
        Arguments passed to the command.
    """
    parser = generate_parser()
    command_line = "{command} {arguments}".format(command=command,
                                                  arguments=" ".join(arguments)
                                                  )
    split_command_line = split(command_line)
    args = parser.parse_args(split_command_line)
    log.debug("Executing command 'gammapy {command}'".format(command=command_line))
    return_code = do_call(args, parser)
    return_code = return_code or 0
    return return_code
