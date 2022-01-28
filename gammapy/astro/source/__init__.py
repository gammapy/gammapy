# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Astrophysical source models."""
from .pulsar import Pulsar, SimplePulsar
from .pwn import PWN
from .snr import SNR, SNRTrueloveMcKee

__all__ = [
    "Pulsar",
    "PWN",
    "SimplePulsar",
    "SNR",
    "SNRTrueloveMcKee",
]
