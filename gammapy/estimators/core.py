# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc

__all__ = ["Estimator"]


class Estimator(abc.ABC):
    """Abstract estimator base class."""

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def _make_selection(self, selection):
        if selection == "all":
            return self.available_selection
        elif selection is None:
            return set()
        elif isinstance(selection, str) and selection in self.available_selection:
            return set([selection])
        else:
            selection = set(selection)
            if selection.issubset(self.available_selection):
                return selection
            else:
                raise ValueError(
                    f"Incorrect selection. Available options are {self.available_selection}"
                )

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, selection):
        self._selection = self._make_selection(selection)