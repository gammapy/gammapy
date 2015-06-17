from __future__ import print_function, division
from gammapy.spectrum import EnergyBinning


__all__ = ['CountsSpectrumDataset',
           ]


class CountsSpectrumDataset(object):
    """
    Counts spectrum dataset (PHA wrapper).
    
    Test case for EnergyBinning class
    """
  
    def __init__(self, counts, energy):

        #does not work (why?) -> Kind of crucial
        #if not isinstance(energy, EnergyBinning):
        #    raise ValueError("energy must be an EnergyBinning object")

        self.energy = energy
        self.counts = counts

        self._check_binning()


    @staticmethod
    def from_fits(hdulist):
        """
        PHA import
        """
        pass

    def to_fits(self):
        """
        PHA export
        """

    def _check_binning(self):
        """
        Check if counts and energy axis match
        """
        if(len(self.counts) != len(self.energy.log_centers)):
            raise ValueError("Counts and energy length do not match")
        
    def info(self):
        """
        Print histogram to screen (not the best solution)
        """

        print("E[{0}]\tCounts".format(self.energy.log_centers.unit))
        for e, c in zip(self.energy.log_centers, self.counts):
            print("{0:.2f}\t{1}".format(e.value,c))
        

    def plot(self):
        """
        Plot counts spectrum
        """
        pass
