# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for progress bar display"""
from contextlib import contextmanager
from tqdm import tqdm


@contextmanager
def pbar(total=None, disable=False):

    if total == None and disable == False:
        raise AttributeError("Can't set up the progress bar if total is None")

    yield tqdm(total=total, mininterval=0, disable=disable)