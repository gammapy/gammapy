# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gammapy: A Python package for gamma-ray astronomy.

* Code: https://github.com/gammapy/gammapy
* Docs: https://docs.gammapy.org/

The top-level `gammapy` namespace is almost empty,
it only contains this:

::

 test                --- Run Gammapy unit tests
 __version__         --- Gammapy version string
 __maincitation__    --- Gammapy main paper
 __acknowledgment__  --- Gammapy acknowledgment


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

__all__ = ["__version__", "song", "__maincitation__", "__acknowledgment__"]

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
del version, PackageNotFoundError


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


# Set the bibtex entry to the article referenced in CITATION.
def _get_bibtex():
    try:
        refs = (
            (Path(__file__).parent.parent / "CITATION")
            .read_text()
            .split("@article")[1:]
        )
        return f"@article{refs[0]}" if refs else ""
    except FileNotFoundError:
        return ""


__maincitation__ = __bibtex__ = _get_bibtex()


def _get_acknowledgment():
    try:
        text = (
            (Path(__file__).parent.parent / "CITATION")
            .read_text()
            .split(
                "If possible, we propose to add the following acknowledgment in LaTeX"
            )
        )
        ackno = text[1].split("\n\n")[1]
        return ackno
    except FileNotFoundError:
        return ""


__acknowledgment__ = _get_acknowledgment()
