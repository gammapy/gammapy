# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Create an illustration of rings used for background estimation
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from gammapy.background import ring


def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size,
                      pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at

fov = 2.0
pixscale = 0.02
areafactor = 20
thetas = [0.1, 0.2, 0.4]
r_is = [0.5, 0.8, 1.1]

x = y = np.arange(-fov, fov + pixscale, pixscale)
X, Y = np.meshgrid(x, y)
d = np.sqrt(X ** 2 + Y ** 2)

fig = plt.figure(1, (10, 10))
title = ('Areafactor = {0} and Pixel Size {1}'
         ''.format(areafactor, pixscale))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(len(thetas), len(r_is)),
                 axes_pad=0.1,
                 )
for i_theta, theta in enumerate(thetas):
    for i_r_i, r_i in enumerate(r_is):
        r_o = ring.r_o(theta, r_i, areafactor)
        circle = d < theta
        ring = (r_i < d) & (d < r_o)
        true_areafactor = ring.sum() / circle.sum()
        mask = circle + ring

        index = i_theta * len(r_is) + i_r_i
        ax = grid[index]
        print index, theta, r_i, r_o, true_areafactor
        ax.imshow(-mask,
                  interpolation='nearest',
                  cmap='gray',
                  extent=[-fov, fov, -fov, fov],
                  )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('r_inner = {0}'.format(r_i))
        ax.set_ylabel('theta = {0}'.format(theta))
        add_inner_title(ax, 'r_thick = {0:1.2f}'.format(r_o - r_i), 2)
        # ax.text(0.05, 0.95, 'r_thick = {0:1.2f}'.format(r_o - r_i),
        #        ha='left', va='top', transform = ax.transAxes)

for extension in ['png', 'pdf']:
    plt.savefig('ringbg_rings_areafactor_{0}_pixscale_{1}.{2}'
                ''.format(areafactor, pixscale, extension), dpi=300)
