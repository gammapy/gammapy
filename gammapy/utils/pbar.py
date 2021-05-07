# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for progress bar display"""
from contextlib import contextmanager

try:
    from tqdm import tqdm
except ImportError:
    class tqdm():
        def __init__(
                self,
                total=0,
                mininterval=0,
                disable=True,
                desc=""
        ):
            pass
        def update(self, x):
            pass

@contextmanager
def pbar(total=None, show_pbar=True, desc=None):
    if total == None and show_pbar == True:
        raise AttributeError("Can't set up the progress bar if total is None")

    yield tqdm(total=total, mininterval=0, disable=not show_pbar, desc=desc)
