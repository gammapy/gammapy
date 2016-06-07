#!/usr/bin/env python

"""script to check the quality of psf fits export.

Usage: pname [options] <directory>
       pname -h | --help
 
       possible plotting styles are semilogx, loglog, default style is linear


Options:
 -r --r68=<double limit>    plot histogram of psf parameterizations where r68 > limit
 -h --help                  Show this message and exit
"""

__author__ = 'dtiziani' 

from astropy.coordinates import Angle

def king_containment_radius(psf, energy, theta, fraction=0.68):
  """Compute containment for all energy and theta values"""

  from astropy.coordinates import Angle
  import astropy.units as u
  from astropy.units import Quantity
  from gammapy.utils.energy import Energy, EnergyBounds
  import scipy.special
  from scipy.optimize import fsolve
  import matplotlib.pyplot as plt

  energy = Energy(energy).flatten()
  theta = Angle(theta).flatten()
  radius = np.empty((theta.size, energy.size))
  for idx_energy in range(len(energy)):
    for idx_theta in range(len(theta)):
      i = np.argmin(np.abs(psf.energy - energy[idx_energy]))
      j = np.argmin(np.abs(psf.offset - theta[idx_theta]))
      
      gamma = float(psf.gamma[j][i])
      sigma = float(psf.sigma[j][i]/(1*u.deg))

      if gamma == 0 or sigma == 0:
        radius[idx_theta][idx_energy] = np.nan
        continue

      print gamma, sigma
      radius[idx_theta][idx_energy] = sigma * np.sqrt(2*gamma*((1-fraction)**(1/(1-gamma))-1)) 
      print  radius[idx_theta][idx_energy]
      

  return Angle(radius, 'deg')

def plot_containment_king(psf, fraction=0.68, ax=None, show_safe_energy=False,
                     add_cbar=True, **kwargs):
  """
  Plot containment image with energy and theta axes.

  Parameters
  ----------
  fraction : float
  Containment fraction between 0 and 1.
  add_cbar : bool
  Add a colorbar
  """
  from matplotlib.colors import PowerNorm
  import matplotlib.pyplot as plt
  ax = plt.gca() if ax is None else ax

  kwargs.setdefault('cmap', 'afmhot')
  kwargs.setdefault('norm', PowerNorm(gamma=0.5))
  kwargs.setdefault('origin', 'lower')
  kwargs.setdefault('interpolation', 'nearest')
  # kwargs.setdefault('vmin', 0.1)
  # kwargs.setdefault('vmax', 0.2)

        # Set up and compute data
  containment = king_containment_radius(psf, psf.energy, psf.offset)

  extent = [
    psf.offset[0].value, psf.offset[-1].value,
    psf.energy[0].value, psf.energy[-1].value,
    ]
  print "Extend: ", containment.T.value
        # Plotting
  ax.imshow(containment.T.value, extent=extent, **kwargs)

  if show_safe_energy:
    # Log scale transformation for position of energy threshold
    e_min = psf.energy.value.min()
    e_max = psf.energy.value.max()
    e = (psf.energy_thresh_lo.value - e_min) / (e_max - e_min)
    x = (np.log10(e * (e_max / e_min - 1) + 1) / np.log10(e_max / e_min)
         * (len(psf.energy_hi) + 1))
    ax.vlines(x, -0.5, len(psf.theta) - 0.5)
    ax.text(x + 0.5, 0, 'Safe energy threshold: {0:3.2f}'.format(psf.energy_thresh_lo))

  # Axes labels and ticks, colobar
  ax.semilogy()
  ax.set_xlabel('Offset (deg)')
  ax.set_ylabel('Energy (TeV)')

  if add_cbar:
    ax_cbar = plt.colorbar(fraction=0.1, pad=0.01, shrink=0.9,
                           mappable=ax.images[0], ax=ax)
    label = 'Containment radius R{0:.0f} (deg)'.format(100 * fraction)
    ax_cbar.set_label(label)

  return ax

def plot_containment_vs_energy_king(psf, fractions=[0.68, 0.95],
                                   thetas=Angle([0, 1], 'deg'), ax=None, **kwargs):
  """Plot containment fraction as a function of energy.
  """
  import matplotlib.pyplot as plt
  from gammapy.utils.energy import Energy, EnergyBounds
  
  ax = plt.gca() if ax is None else ax

  energy = Energy.equal_log_spacing(
    psf.energy[0], psf.energy[-1], 100)

  for theta in thetas:
    for fraction in fractions:
      radius = king_containment_radius(psf, energy, theta, fraction).squeeze()
      label = '{} deg, {:.1f}%'.format(theta, 100 * fraction)
      ax.plot(energy.value, radius.value, label=label)

  ax.semilogx()
  ax.legend(loc='best')
  ax.set_xlabel('Energy (TeV)')
  ax.set_ylabel('Containment radius (deg)')

def peek_king(psf, figsize=(15, 5)):
  """Quick-look summary plots."""
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)

  plot_containment_king(psf, fraction=0.68, ax=axes[0])
  plot_containment_king(psf, fraction=0.95, ax=axes[1])
  plot_containment_vs_energy_king(psf, ax=axes[2])

  # TODO: implement this plot
  # psf = self.psf_at_energy_and_theta(energy='1 TeV', theta='1 deg')
  # psf.plot_components(ax=axes[2])

  plt.tight_layout()
  plt.show()

def plot_containment_table(filename, theta, fraction=0.68, ax=None, **kwargs):
  from gammapy.irf import EnergyDependentTablePSF
  from matplotlib.colors import PowerNorm
  import numpy as np
  radii = np.zeros((6,18))
  extent = None
  for i in range(theta.size):
    psf = EnergyDependentTablePSF.read_psf_table(filename, theta[i])
    a = np.zeros(psf.energy.size)
    extent = [
      theta[0].value, theta[-1].value,
      psf.energy[0].to('TeV').value, psf.energy[-1].to('TeV').value,
      ]
    
    for e in range(psf.energy.size):
      print "Next"
      skip = False
      for v in psf.psf_value[e]:
        if np.isnan(v) or v.value == 0:
          skip = True
          break
      if skip:
        continue
      try:
        a[e] = psf.containment_radius(psf.energy[e], fraction).degree
        print theta[i], a[e]
      except ValueError:
        a[e] = 0
    radii[i]=a

  for x in np.nditer(radii, op_flags=['readwrite']):
    if x == 0:
      x[...] = np.max(radii)
  
  import matplotlib.pyplot as plt
  
  ax = plt.gca() if ax is None else ax

  kwargs.setdefault('cmap', 'afmhot')
  kwargs.setdefault('norm', PowerNorm(gamma=0.5))
  kwargs.setdefault('origin', 'lower')
  kwargs.setdefault('interpolation', 'nearest')

  
  ax.imshow(radii.T,extent=extent, **kwargs)

  ax.semilogy()
  ax.set_xlabel('Offset (deg)')
  ax.set_ylabel('Energy (TeV)')

  ax_cbar = plt.colorbar(fraction=0.1, pad=0.01, shrink=0.9,
                           mappable=ax.images[0], ax=ax)
  label = 'Containment radius R{0:.0f} (deg)'.format(100 * fraction)
  ax_cbar.set_label(label)
  
def plot_containment_vs_energy_table(filename, fractions=[0.68, 0.95],
                                   thetas=Angle([0, 1], 'deg'), ax=None, **kwargs):
  """Plot containment fraction as a function of energy.
  """
  import matplotlib.pyplot as plt
  from gammapy.utils.energy import Energy, EnergyBounds
  from gammapy.irf import EnergyDependentTablePSF
  
  ax = plt.gca() if ax is None else ax



  for theta in thetas:
    psf = EnergyDependentTablePSF.read_psf_table(filename, theta)
    energy = Energy.equal_log_spacing(
      psf.energy[0], psf.energy[-1], 100)
    for fraction in fractions:
      radius = np.zeros(energy.size)
      for e in range(energy.size):
        skip = False
        for v in psf.psf_value[e]:
          if np.isnan(v) or v.value == 0:
            skip = True
            break
        if skip:
          continue
        radius[e] = psf.containment_radius(energy[e], fraction).degree
      label = '{} deg, {:.1f}%'.format(theta, 100 * fraction)
      ax.plot(energy.value, radius.value, label=label)

  ax.semilogx()
  ax.legend(loc='best')
  ax.set_xlabel('Energy (TeV)')
  ax.set_ylabel('Containment radius (deg)')
  
def peek_table(filename, theta, figsize=(15, 5)):
  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

  plot_containment_table(filename, theta, fraction=0.68, ax=axes[0])
  plot_containment_table(filename, theta, fraction=0.95, ax=axes[1])
  plot_containment_table(filename, theta, fraction=0.99, ax=axes[2])
  #plot_containment_vs_energy_table(filename, ax=axes[2])

  plt.tight_layout()
  plt.show()
    

##############################################################################################
# Parse command-line arguments
##############################################################################################
import docopt
args = docopt.docopt(__doc__)

#import
import numpy as np
import os
from astropy.io import fits
import docopt
import matplotlib.pyplot as plt
from gammapy.data import DataStore
from gammapy import irf


#get runlist
dir = args['<directory>']

#check if file exists/if file is in fits format
if not os.path.exists(dir):
  sys.exit("Directory not found.")


##############################################################################################
# main routine
##############################################################################################


data_store = DataStore.from_dir(dir)
obs_ids = data_store.obs_table['OBS_ID'].data
print 'Number of observations: ', obs_ids.size

for obs_id in obs_ids:
    print obs_id
    psf_gauss = data_store.obs(obs_id=obs_id).load(hdu_class='psf_3gauss')
    psf_king = data_store.obs(obs_id=obs_id).load(hdu_class='psf_king')
    
    #plot_table_containment(data_store.obs(obs_id=obs_id).location(hdu_class="psf_table").path(), psf_gauss.theta)

    peek_table('run018400-018599/run018417/hess_psf_table_018417.fits.gz', psf_gauss.theta)

    break


    radii = psf_gauss.containment_radius(psf_gauss.energy_hi, psf_gauss.theta)
    #radii2 = king_containment_radius(psf_king, psf_king.energy, psf_king.offset)
    over_limits = np.zeros(radii.shape)
    differences = np.zeros(radii.shape)
#    peek_king(psf_king)
    psf_gauss.peek()
    for t in range(radii.shape[0]):
        for e in range(radii.shape[1]):
            #Check for bounds
            try:
                if (radii[t][e] > Angle(0.2, 'deg')):
                    over_limits[t][e] = 1
            except ValueError:
                print 'error'
            #Check for differences
            nb_t = np.array([t])
            nb_e = np.array([e])
            if t+1 < radii.shape[0]:
                nb_t = np.append(nb_t, t+1)
            if t > 0:
                nb_t = np.append(nb_t, t-1)
            if e+1 < radii.shape[1]:
                nb_e = np.append(nb_e, e+1)
            if e > 0:
                nb_e = np.append(nb_e, e-1)
            for t_i in nb_t:
                for e_i in nb_e:
                    if abs(radii[t][e]-radii[t_i][e_i])/radii[t][e] > 0.2:
                        differences[t][e] = 1

    print over_limits
    print differences
    print psf_king
    
    print psf_gauss.theta
