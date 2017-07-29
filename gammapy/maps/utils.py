# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals


def unpack_seq(seq, n=1):
    """Utility to unpack the first N values of a tuple or list.  Remaining
    values are put into a single list which is the last element of the
    return value.  This partially simulates the extended unpacking
    functionality available in Python 3.

    Parameters
    ----------
    seq : list or tuple
        Input sequence to be unpacked.

    n : int
        Number of elements of ``seq`` to unpack.  Remaining elements
        are put into a single tuple.

    """

    for row in seq:
        yield [e for e in row[:n]] + [row[n:]]
