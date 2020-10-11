# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for progress bar display"""
from contextlib import contextmanager
from tqdm.auto import tqdm


@contextmanager
def pbar(total=None, show_pbar=False):
    class dummy():
        def update(self, x):
            pass

    if total == None and show_pbar == True:
        raise AttributeError("Can't set up the progress bar if total is None")

    yield tqdm(total=total, mininterval=0) if show_pbar == True else dummy()
