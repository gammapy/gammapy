# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for progress bar display"""
import logging

log = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except ImportError:

    class tqdm:
        def __init__(self, iterable, disable=True, **kwargs):
            self.disable = disable
            self._iterable = iterable
            if not self.disable:
                log.info(
                    "Tqdm is currently not installed. Visit https://tqdm.github.io/"
                )

        def update(self, x):
            pass

        def __iter__(self):
            return self._iterable.__iter__()

        def __next__(self):
            return self._iterable.__next__()


SHOW_PROGRESS_BAR = False


def progress_bar(iterable, desc=None):
    # Necessary because iterable may be a zip
    iterable_to_list = list(iterable)
    total = len(iterable_to_list)

    return tqdm(
        iterable_to_list,
        total=total,
        mininterval=0,
        disable=not SHOW_PROGRESS_BAR,
        desc=desc,
    )
