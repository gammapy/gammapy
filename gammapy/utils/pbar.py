# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for progress bar display"""
import logging
from collections.abc import Iterable

log = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm():
        def __init__(self, iterable, disable=True, **kwargs):
            self.disable = disable
            self._iterable = iterable
            if self.disable == False:
                log.info(
                    f"Tqdm is currently not installed. Visit https://tqdm.github.io/"
                )

        def update(self, x):
            pass

        def __iter__(self):
            return self._iterable.__iter__()

        def __next__(self):
            return self._iterable.__next__()

def progress_bar(iterable, show_progress_bar=False, desc=None):
    if not isinstance(iterable, Iterable) and show_progress_bar == True:
        raise AttributeError("Can't set up the progress bar if total is None")

    # Necessary because iterable may be a zip
    total = len(list(iterable))

    return tqdm(iterable, total=total, mininterval=0, disable=not show_progress_bar, desc=desc)