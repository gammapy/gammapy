# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for progress bar display"""
from contextlib import contextmanager

try:
    from tqdm import tqdm
except ImportError:
    class tqdm():
        def __init__(self, *args, **kwargs):
            pass
        def update(self, x):
            pass

@contextmanager
def pbar(total=None, show_progress_bar=True, desc=None):
    if total is None and show_progress_bar == True:
        raise AttributeError("Can't set up the progress bar if total is None")

    yield tqdm(total=total, mininterval=0, disable=not show_progress_bar, desc=desc)
