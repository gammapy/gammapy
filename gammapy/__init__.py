# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gammapy: A Python package for gamma-ray astronomy.

* Code: https://github.com/gammapy/gammapy
* Docs: https://docs.gammapy.org/

The top-level `gammapy` namespace is almost empty,
it only contains this:

::

 test              --- Run Gammapy unit tests
 __version__       --- Gammapy version string


The Gammapy functionality is available for import from
the following sub-packages (e.g. `gammapy.makers`):

::

 astro        --- Astrophysical source and population models
 catalog      --- Source catalog tools
 makers       --- Data reduction functionality
 data         --- Data and observation handling
 irf          --- Instrument response functions (IRFs)
 maps         --- Sky map data structures
 modeling     --- Models and fitting
 estimators   --- High level flux estimation
 stats        --- Statistics tools
 utils        --- Utility functions and classes
"""

__all__ = ["__version__", "song"]

import pkg_resources

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    # package is not installed
    pass


def song(karaoke=False):
    """
    Listen to the Gammapy song.

    Make sure you listen on good headphones or speakers. You'll be not disappointed!

    Parameters
    ----------
    karaoke : bool
        Print lyrics to sing along.
    """
    import sys
    import webbrowser

    webbrowser.open("https://gammapy.org/gammapy_song.mp3")

    if karaoke:
        lyrics = (
            "\nGammapy Song Lyrics\n"
            "-------------------\n\n"
            "Gammapy, gamma-ray data analysis package\n"
            "Gammapy, prototype software CTA science tools\n\n"
            "Supernova remnants, pulsar winds, AGN, Gamma, Gamma, Gammapy\n"
            "Galactic plane survey, pevatrons, Gammapy, Gamma, Gammapy\n"
            "Gammapy, github, continuous integration, readthedocs, travis, "
            "open source project\n\n"
            "Gammapy, Gammapy\n\n"
            "Supernova remnants, pulsar winds, AGN, Gamma, Gamma, Gammapy\n"
        )

        centered = "\n".join(f"{s:^80}" for s in lyrics.split("\n"))
        sys.stdout.write(centered)
