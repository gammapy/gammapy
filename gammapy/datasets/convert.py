# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Convert dataset formats.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from ..obs import ObservationTable

__all__ = ['convert_obs_list_format_to_gammapy',
           'convert_obs_list_hess_to_gammapy',
           ]


def convert_obs_list_format_to_gammapy(obs_list, scheme):
    """Convert oservation list from supported formats to Gammapy format.

    This script calls the corresponding format converter, depending
    on the value of the **scheme** parameter, in order to make
    observation lists from different experiments comply to the format
    described in :ref:`dataformats_observation_lists`.

    Curretly only the H.E.S.S. scheme is supported, which follows the
    guidelines dictated for CTA.

    Parameters
    ----------
    obs_list : `~astropy.table.Table`
        Observation list to convert.
    scheme : str
        Format of the input observation list to convert.

    Returns
    -------
    obs_table : `~gammapy.obs.ObservationTable`
        Converted observation list.
    """
    if scheme == 'hess':
        return convert_obs_list_hess_to_gammapy(obs_list)
    else:
        raise ValueError('Invalid scheme: {}'.format(scheme))


def convert_obs_list_hess_to_gammapy(obs_list):
    """Convert oservation list from H.E.S.S./CTA format to Gammapy format.

    The H.E.S.S. observation lists are produced following the
    guidelines dictated for CTA. This function should convert the
    format to the one described in :ref:`dataformats_observation_lists`.

    This script renames the columns and edits the header keywords of
    the observation lists. Columns and header keywords not defined in
    :ref:`dataformats_observation_lists` are left unchanged.

    This function has no tests implemented, since the H.E.S.S. data
    is private.

    Parameters
    ----------
    obs_list : `~astropy.table.Table`
        Observation list to convert.

    Returns
    -------
    obs_table : `~gammapy.obs.ObservationTable`
        Converted observation list.
    """
    obs_table = ObservationTable(obs_list)

    # rename column names
    renames = [('RA_PNT', 'RA'),
               ('DEC_PNT', 'DEC'),
               ('ALT_PNT', 'ALT'),
               ('AZ_PNT', 'AZ'),
               ('MUONEFF', 'MUON_EFFICIENCY'),
               ('ONTIME', 'TIME_OBSERVATION'),
               ('LIVETIME', 'TIME_LIVE'),
               ('TSTART', 'TIME_START '),
               ('TSTOP', 'TIME_STOP'),
               ('TRGRATE', 'TRIGGER_RATE'),
               ('MEANTEMP', 'MEAN_TEMPERATURE'),
               ('TELLIST', 'TEL_LIST')
               ]
    for name, new_name in renames:
        obs_table.rename_column(name, new_name)

    # add missing header entries
    obs_table.meta['OBSERVATORY_NAME'] = 'HESS'
    obs_table.meta['TIME_FORMAT'] = 'relative'

    return obs_table
