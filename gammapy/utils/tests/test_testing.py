# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.utils.testing import assert_str_equal


def test_assert_str_equal():
    assert_str_equal("a\nb\n", "a\nb\n")
    assert_str_equal("Class : PSR  \nName : x\n", "Class : PSR\nName : x   \n")


def test_assert_str_equal_fails():
    with pytest.raises(AssertionError):
        assert_str_equal("Class : PSR\n", "Class : SNR\n")

    with pytest.raises(AssertionError):
        assert_str_equal("  a\n", "a\n")

    with pytest.raises(AssertionError):
        assert_str_equal("a\nb\n", "a\n")
