"""
Plot Fermi IRFs similar to what is available here:
http://www-glast.slac.stanford.edu/software/IS/glast_lat_performance.htm

Jeremy Perkins, Christoph Deil
Based on plot_irfs.py by Jim Chiang which is distributed with the ST.

TODO:
- 95% / 68% front ratio at 1e5 MeV is 2.85, but on Fermi Performance page is 4.5
- Compute integral for containment fraction in a better, faster way
- Effective area lookup has strange wiggles not present in the Fermi plots
- Energy resolution lookup and 68% containment calculation
- Plots for zenith angle dependence
- Clean up the interface (no global variables)
- Implement sensitivity computation
"""
import numpy as np
import matplotlib
matplotlib.use('pdf')
#import matplotlib.pyplot as plt
#import pylab
#from pylab import *

import pyIrfLoader
pyIrfLoader.Loader_go()

class irf(object):
    _factory = pyIrfLoader.IrfsFactory_instance()
    def __init__(self, irfsName, inc=0, phi=0):
        self._irfs = self._factory.create(irfsName)
        self._inc = inc
        self._phi = phi

class Psf(irf):
    """Parameters: energy, inc, phi in constructor and separation in __call__"""
    def __init__(self, irfsName, energy=1e3, inc=0, phi=0):
        irf.__init__(self, irfsName, inc, phi)
        self._psf = self._irfs.psf()
        self._energy = energy
    def __call__(self, separation):
        psf, energy, inc, phi = self._psf, self._energy, self._inc, self._phi
        try:
            y = []
            for x in separation:
                y.append(psf.value(x, energy, inc, phi))
            return np.array(y)
        except TypeError:
            return psf.value(separation, energy, inc, phi)
    def containment_angle(self, containment_fraction=0.68):
        """Compute the angle that contains a certain fraction of events"""
        sep = np.logspace(-3, 2, 1e4)
        dn_dsep = self.__call__(sep) # number of photons per deg
        dn = dn_dsep[:-1] * np.diff(sep) # number of photons per bin
        containment = dn.cumsum() / dn.sum() # fraction of photons at smaller angles
        
        bin = np.where(containment > containment_fraction)[0][0]
        angle = sep[bin]
        #print 'Containment bin', bin, 'at angle', angle
        return angle

def psf_containment(energies, irfsName, 
                    containment_fraction=0.68,
                    inc=0, phi=0):
    """Compute containment angle for front or back"""
    angles = []
    for energy in energies:
        psf_front = Psf(irfsName, energy)
        angles.append(psf_front.containment_angle(containment_fraction))
    return np.array(angles)

def psf_containments(energy, irf_name, containment_fraction):
    """Compute containment angle for front, back and total"""
    # To compute the total angle we need the effective area-weighted,
    # which is equal to the count-weighted average
    aeff_front = Aeff(irf_name + '::FRONT')(energy) 
    aeff_back = Aeff(irf_name + '::BACK')(energy) 
    aeff_total = aeff_front + aeff_back 

    angle_front = psf_containment(energy, irf_name + '::FRONT',
                                  containment_fraction)
    angle_back = psf_containment(energy, irf_name + '::BACK',
                                 containment_fraction)
    angle_total = (aeff_front * angle_front + 
                   aeff_back * angle_back) / aeff_total
    return angle_front, angle_back, angle_total

class Aeff(irf):        
    """Parameters: inc and phi in constructor and energy in __call__"""
    def __init__(self, irfsName, inc=0, phi=0):
        irf.__init__(self, irfsName, inc, phi)
        self._aeff = self._irfs.aeff()
    def __call__(self, energy):
        aeff, inc, phi = self._aeff, self._inc, self._phi
        try:
            y = []
            for x in energy:
                #print x, inc, phi
                y.append(aeff.value(x, inc, phi))
            return np.array(y)
        except TypeError:
            return aeff.value(energy, inc, phi)

# Parameters for plots
energy = np.logspace(2, 6, 1000) # x-values for effective area
sep = np.logspace(-3, 0, 1000) # x-values for PSF
inc = np.linspace(0, 90, 1000) # x-values for inclination
irf_name = 'P6_V3_DIFFUSE'
irf_name = 'P6_V3_DATACLEAN'

def plot_aeff(filename='fermi_aeff_energy_dependence.pdf'):
    """Plot on-axis effective area as a function of energy"""
    aeff_front = Aeff(irf_name + '::FRONT')(energy) 
    aeff_back = Aeff(irf_name + '::BACK')(energy) 
    aeff_total = aeff_front + aeff_back 
    
    clf()
    semilogx()
    plot(energy, aeff_front, color='red', label='front')
    plot(energy, aeff_back, color='blue', label='back')
    plot(energy, aeff_total, color='black', label='total')
    xlabel('Energy (MeV)')
    ylabel('Effective Area (cm^2)')
    grid(True)
    legend()
    savefig(filename)
    
def plot_psf(filename='fermi_psf_sep_dependence.pdf'):
    """Plot on-axis PSF a function of separation for a bunch of energy"""
    psf_front = Psf(irf_name + '::FRONT', energy=1e6)(sep) 
    psf_back = Psf(irf_name + '::BACK', energy=1e6)(sep)
    # TODO: weigh by effective area
    psf_total = psf_front + psf_back 
    
    clf()
    #semilogy()
    plot(sep, psf_front, color='red', label='front')
    plot(sep, psf_back, color='blue', label='back')
    plot(sep, psf_total, color='black', label='total')
    xlabel('Separation (deg)')
    ylabel('PSF (arbitrary units)')
    grid(True)
    legend()
    savefig(filename)

def plot_psf_containment():
    """Plot on-axis PSF containment angle as a function of energy"""
    angle_front_68, angle_back_68, angle_total_68 = \
    psf_containments(energy, irf_name, 0.68)

    angle_front_95, angle_back_95, angle_total_95 = \
    psf_containments(energy, irf_name, 0.95)
    
    # Plot containment fraction for 68 and 95
    clf()
    loglog()
    plot(energy, angle_front_68, 'r', label='front 68%')
    plot(energy, angle_back_68, 'b', label='back 68%')
    plot(energy, angle_total_68, 'k', label='total 68%')
    plot(energy, angle_front_95, 'r--', label='front 95%')
    plot(energy, angle_back_95, 'b--', label='back 95%')
    plot(energy, angle_total_95, 'k--', label='total 95%')
    xlabel('Energy (MeV)')
    ylabel('Containment Angle (deg)')
    grid(True)
    legend()
    savefig('fermi_psf_containment_energy_dependence.pdf')

    # Plot containment fraction ratio 95 / 68
    clf()
    semilogx()
    plot(energy, angle_front_95 / angle_front_68, 'r', label='front')
    plot(energy, angle_back_95 / angle_back_68, 'b', label='back')
    plot(energy, angle_total_95 / angle_total_68, 'k', label='total')
    xlabel('Energy (MeV)')
    ylabel('95% / 68% Containment Angle Ratio')
    grid(True)
    legend()
    savefig('fermi_psf_containment_ratio_energy_dependence.pdf')

if __name__ == '__main__':
    plot_aeff()
    #plot_psf()
    #plot_psf_containment()