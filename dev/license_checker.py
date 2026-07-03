# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This script check that all the file in a given folder and its subfolder start with
the license statement. Can also be applied to single file. If the `--fix` or `-f`
option is used, fix by prepending the license statement at the beginning of the file.

Usage:
  python license_checker <PATH> [Options]
  Options:
    --fix, -f:  Fix by prepending the license statement to the top of the file
    --help, -h: Display this message
"""

import pathlib
from dataclasses import dataclass
from typing import Sequence
import tempfile
import os
import io


LICENSE_STATEMENT = "# Licensed under a 3-clause BSD style license - see LICENSE.rst"


TAB = "  "


RESERVED_FIRST_LINE = ("#!", "# -*- coding: utf-8 -*-")


class NoArgumentError(Exception):
    pass


@dataclass
class PrintColor:
    WARNGING: str = "\033[93m"
    ENDC: str = "\033[0m"
    FAIL: str = "\033[91m"
    OKBLUE: str = "\033[94m"


def print_usage():
    print("Usage:")
    print(f"{TAB}python license_checker <PATH> [Options]")
    print(f"{TAB}Options:")
    print(
        f"{TAB}{
            TAB
        }--fix, -f:  Fix by prepending the license statement to the top of the file"
    )
    print(f"{TAB}{TAB}--help, -h: Display this message")


def prepend_line(
    file_path: pathlib.Path,
    original_file: io.TextIOWrapper,
    reserved_first_line: bool = False,
):
    """
    Prepend the license statement to a file. Uses `tempfile` to prevent data losses.

    Parameters
    ----------
        file_path: `pathlib.Path`
            Path to the file.
        original_file: `io.TextIOWrapper`
           Instance of the open original file.
        reserved_first_line: str, Optional
            Whether the original file starts with a reserved first line, e.g. shebang
            (`#!`) for executable or `# -*- coding: utf-8 -*-` in docs python file.
            If that is the case, put the license statement on the second line.
            Default if False.
    """

    file_descriptor, tmp_path = tempfile.mkstemp(dir=file_path.parent)

    with os.fdopen(file_descriptor, "w") as tmp_file:
        if reserved_first_line:
            tmp_file.write(original_file.readline())
        tmp_file.write(LICENSE_STATEMENT + "\n")
        tmp_file.writelines(original_file)
    os.replace(tmp_path, file_path)


def check_and_fix(file: pathlib.Path, fix: bool = False):
    """
    Check that the file start with the license statement. Fix it if the `fix`
    parameter is set to `True`.

    Parameters
    ----------
    file: pathlib.Path
        Path to the file.
    fix: bool, Optional
        Whether to fix by prepending the license statement to the file.
        Default is False.
    """
    reserved_first_line = False
    with open(file, "r") as f:
        line = f.readline().strip()
        if line.startswith(RESERVED_FIRST_LINE):
            reserved_first_line = True
            line = f.readline().strip()
        if line != LICENSE_STATEMENT:
            print(
                f"{PrintColor.FAIL}{file}{
                    PrintColor.ENDC
                } does not start with the license statement !"
            )
            print(f"Start with:\n\t{PrintColor.WARNGING}{line}{PrintColor.ENDC}")
            if fix:
                print(f"{PrintColor.OKBLUE}Fixing...{PrintColor.ENDC}\n")
                _ = f.seek(0)
                prepend_line(file, f, reserved_first_line)
            else:
                print()


def main(argv: Sequence[str] | None = None):
    try:
        if argv[0] in ["--help", "-h"]:
            print_usage()
            exit()
        root_path = pathlib.Path(argv[0])
    except IndexError:
        raise NoArgumentError("Missing required argument: path to apply this script")

    fix = False
    if (len(argv) == 2) and (argv[1] in ["--fix", "-f"]):
        fix = True
    if root_path.is_file():
        check_and_fix(root_path, fix)

    files = map(pathlib.Path, root_path.rglob("*.py"))

    for file in files:
        check_and_fix(file, fix)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
