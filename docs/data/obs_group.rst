.. _obs_observation_grouping:

Observation grouping
====================

Observation grouping can be very helpful to speed the analysis of
data with similar characteristics. It is also essential in some cases,
like the production of background models. In this case, the scarce
statistic forces the grouping of observations taken under similar
conditions in order to produce reliable models.

Compute groups
--------------

Observation grouping can be done by taking a `astropy.table.Table`
with observation parameters (with colum ``OBS_ID``, but also e.g.
``ZENITH`` or ``N_TELS``), and using some grouping method to assign
an integer ``GROUP_ID`` to each.

As an example, let's show how to group by two parameters:

- ``ZENITH`` bins with edges ``[0, 30, 40, 50, 90]`` degrees
- ``N_TELS`` bins with values ``[3, 4]``

This gives a total of eight groups (with ``GROUP_ID = 0 .. 7``),
four on the ``ZENITH`` axis and two on the ``N_TELS`` axis.

We use `numpy.digitize` to compute the group index along each axis,
and then `numpy.ravel_multi_index` to create a single ``OBS_ID`` from
the two axis group indices.


.. code-block:: python

    import numpy as np
    from astropy.table import Table

    table = Table.read('https://github.com/cdeil/HESS-DL3-DR1-preview/raw/master/obs-index.fits.gz')

    zen_pnt_bins = np.array([0, 30, 40, 50, 90])
    zen_pnt_idx = np.digitize(table['ZEN_PNT'].data, bins=zen_pnt_bins) - 1

    n_tels = np.array([3, 4])
    n_tels_idx = np.digitize(table['N_TELS'].data, bins=n_tels) - 1

    group_id = np.ravel_multi_index(
        multi_index=(zen_pnt_idx, n_tels_idx),
        dims=(len(zen_pnt_bins) - 1, len(n_tels)),
    )

    table['GROUP_ID'] = group_id

Note that this is just an example; with a few lines of Python and Numpy,
you can compute any grouping you like. You can do something simple like
what we did here, or you could do something very fancy, such as using
a cluster method from scikit-learn to cluster observations by certain parameters.
Another option is to compute IRF characteristics (e.g. similar energy threshold)
for each observation, and then group based on that. Usually grouping is done
before stacking runs within a given group and you want their IRFs to be similar.

Use groups
----------

We currently don't use ``GROUP_ID`` in observation tables within Gammapy.
For now, it's left up to the user to group observations and usually compute
one set of stacked maps or spectra per group, and then to use those in a joint
likelihood fit. We might or might not build this grouping functionality into
Gammapy, e.g. by using ``GROUP_ID`` if present and stacking observations within
each group in the spectrum or map analysis.

For now, for spectra, it's already possible to group observations like this:

.. code-block:: python

    import numpy as np
    table = "Table with GROUP_ID column, see above"
    spectra = []
    for group_id in np.unique(table['GROUP_ID'].data):
        group_table = table[table['GROUP_ID'] == group_id]
        obs_id = group_table['OBS_ID']
        # Make a stacked spectrum for these `group_id`
        spectra.append(spectrum)

    # Pass list of grouped spectra to SpectrumFit for joint fit

TODO: make a complete, fully working example how to do a grouped analysis
