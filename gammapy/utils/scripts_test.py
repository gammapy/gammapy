# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test if Jupyter notebooks work."""
import os
import sys
import logging
from pathlib import Path
from pkg_resources import working_set
import subprocess
import yaml

log = logging.getLogger(__name__)


def get_scripts():
    """Read `scripts.yaml` info."""
    path = Path("examples") / "scripts.yaml"
    with path.open() as fh:
        return yaml.safe_load(fh)


def requirement_missing(script):
    """Check if one of the requirements is missing."""
    if "requires" in script:
        if script["requires"] is None:
            return False
        for package in script["requires"].split():
            try:
                working_set.require(package)
            except Exception:
                return True
    return False


def script_test(path):
    """Check if example Python script is broken."""
    log.info("   ... EXECUTING {}".format(str(path)))

    cmd = ["python", str(path)]
    cp = subprocess.run(cmd, capture_output=True)
    if cp.returncode:
        log.info("   ... FAILED")
        log.info("   ___ TRACEBACK")
        log.info(cp.stderr.decode('utf-8')+"\n\n")
        return False
    else:
        log.info("   ... PASSED")
        return True


def main():
    logging.basicConfig(level=logging.INFO)

    if "GAMMAPY_DATA" not in os.environ:
        log.info("GAMMAPY_DATA environment variable not set.")
        log.info("Running scripts tests requires this environment variable.")
        log.info("Exiting now.")
        sys.exit()

    passed = True

    for script in get_scripts():
        if requirement_missing(script):
            log.info(
                "Skipping script {} because requirement is missing.".format(
                    script["name"]
                )
            )
            continue

        filename = script["name"] + ".py"
        path = Path("examples") / filename

        if not script_test(path):
            passed = False

    assert passed


if __name__ == "__main__":
    main()
