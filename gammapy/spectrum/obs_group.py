# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.coordinates import Angle
from astropy.table import Column
from astropy.units import Quantity
from ..extern.pathlib import Path
from ..data import ObservationGroupAxis, ObservationGroups, ObservationTable
from ..spectrum import SpectrumObservationList, SpectrumObservation

# TODO: remove this whole file?
# Replace it with docs how to do something like this using just Astropy Table or pandas DataFrame?

__all__ = [
    'SpectrumObservationGrouping',
    'group_obs_table',
]

log = logging.getLogger(__name__)


def group_obs_table(obs_table, offset_range=[0, 2.5], n_off_bin=5,
                    eff_range=[0, 100], n_eff_bin=4, zen_range=[0., 70.],
                    n_zen_bin=7):
    """Helper function to provide an observation grouping in offset,
    muon_efficiency, and zenith.

    Parameters
    ----------
    obs_table : `~gammapy.data.ObservationTable`
        Obs table to group
    offset_range : tuple
        Range of the offset band
    n_off_bin : int
        Number of offset bins
    eff_range : tuple
        Range of the muon efficiency band
    n_eff_bin : int
        Number of muon efficiency bins
    zen_range : tuple
        Range of the zenith angle band
    n_zen_bin : int
        Number of zenith bins

    Returns
    -------
    grouped_table : `~gammapy.data.ObservationTable`
    """
    offmin, offmax = offset_range
    effmin, effmax = eff_range
    zenmin, zenmax = zen_range
    offtab = Angle(np.linspace(offmin, offmax, n_off_bin + 1), 'deg')
    efftab = Quantity(np.linspace(effmin, effmax, n_eff_bin + 1) / 100., '')
    zentab = Quantity(np.linspace(zenmin, zenmax, n_zen_bin + 1), 'deg')
    coszentab = np.cos(zentab)[::-1]

    val = list()
    val.append(ObservationGroupAxis('MUONEFF', efftab, 'edges'))
    val.append(ObservationGroupAxis('COSZEN', coszentab, 'edges'))
    val.append(ObservationGroupAxis('OFFSET', offtab, 'edges'))

    obs_groups = ObservationGroups(val)
    cos_zen = np.cos(obs_table['ZEN_PNT'].quantity)
    obs_table.add_column(Column(cos_zen, 'COSZEN'))
    grouped_table = obs_groups.apply(obs_table)

    return grouped_table


class SpectrumObservationGrouping(object):
    """
    Class for stacking observations in groups

    The format of the input observation table
    is described in :ref:`dataformats_observation_lists`. The column
    ``PHAFILE`` is added after a `~gammapy.spectrum.SpectrumExtraction`. The
    column ``GROUP_ID`` can be added as described in
    :ref:`obs_observation_grouping`.

    Parameters
    ----------
    obs_table : `~gammapy.spectrum.SpectrumObservationList`
        Observation table with group ID column
    """

    def __init__(self, obs_table):
        self.obs_table = obs_table
        self.stacked_observations = None
        self.stacked_obs_table = None

    def stack_groups(self):
        """Stack observations in each group."""
        stacked_obs = list()

        sorted_table = self.obs_table.group_by('GROUP_ID')
        for group in sorted_table.groups:
            group_id = group['GROUP_ID'][0]
            log.info('Stacking observations in group {}'.format(group_id))
            log.info('{}'.format([group['OBS_ID']]))
            temp = SpectrumObservationList.from_observation_table(group)
            stacked = SpectrumObservation.stack_observation_list(temp)
            stacked.meta.phafile = 'pha_group{}.fits'.format(group_id)
            stacked.meta.ogip_dir = Path.cwd() / 'ogip_data_stacked'

            stacked_obs.append(stacked)

        self.stacked_observations = SpectrumObservationList(stacked_obs)

    def make_observation_table(self):
        """Create observation table for the stacked observations."""
        phafile = [str(o.meta.ogip_dir / o.meta.phafile) for o in self.stacked_observations]
        col1 = Column(data=phafile, name='PHAFILE')
        # Todo: Put meta information about the groups in the table
        self.stacked_obs_table = ObservationTable([col1])

    def write(self):
        """Write stacked observations and observation table."""
        self.stacked_observations.write_ogip()
        self.stacked_obs_table.write('observation_table_stacked.fits',
                                     overwrite=True)

    def run(self):
        """Run all steps."""
        self.stack_groups()
        self.make_observation_table()
        self.write()
