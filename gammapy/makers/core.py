# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import numpy as np

__all__ = ["Maker"]


class Maker(abc.ABC):
    """Abstract maker base class."""

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def __str__(self):
        s = f"{self.__class__.__name__}\n"
        s += "-" * (len(s) - 1) + "\n\n"

        names = self.__init__.__code__.co_varnames

        max_len = np.max([len(_) for _ in names]) + 1

        for name in names:
            value = getattr(self, name, None)

            if value is None:
                continue
            else:
                s += f"\t{name:{max_len}s}: {value}\n"

        return s.expandtabs(tabsize=2)
