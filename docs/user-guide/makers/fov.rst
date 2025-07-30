.. include:: ../../references.txt

.. _fov_background:

**************
FoV background
**************

Overview
--------

Background models stored in IRF might not predict accurately the actual number of background counts.
To correct the predicted counts, one can use the data themselves in regions deprived of gamma-ray signal.
The field-of-view background technique is used to adjust the predicted counts on the measured ones outside
an exclusion mask. This technique is recommended for 3D analysis, in particular when stacking `~gammapy.datasets.Datasets`.

Gammapy provides the `~gammapy.makers.FoVBackgroundMaker`. The latter creates a
`~gammapy.modeling.models.FoVBackgroundModel` which combines the ``background`` predicted number of counts
and a ``NormSpectralModel`` which allows to renormalize the background cube, and
possibly to change its spectral distribution. By default, only the ``norm`` parameter of a
`~gammapy.modeling.models.PowerLawNormSpectralModel` is left free. Here we show the addition of a
`~gammapy.modeling.models.PowerLawNormSpectralModel`
in which the ``norm`` and ``tilt`` parameters are unfrozen as an example.

.. testcode::

	from gammapy.makers import MapDatasetMaker, FoVBackgroundMaker, SafeMaskMaker
	from gammapy.datasets import MapDataset
	from gammapy.data import DataStore
	from gammapy.modeling.models import PowerLawNormSpectralModel
	from gammapy.maps import MapAxis, WcsGeom, Map
	from regions import CircleSkyRegion
	from astropy import units as u

	data_store = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
	observations = data_store.get_observations([23592, 23559])

	energy_axis = MapAxis.from_energy_bounds("0.5 TeV", "10 TeV", nbin=5)
	energy_axis_true = MapAxis.from_energy_bounds("0.3 TeV", "20 TeV", nbin=20, name="energy_true")

	geom = WcsGeom.create(skydir=(83.63, 22.01), axes=[energy_axis], width=5, binsz=0.02)

	stacked = MapDataset.create(geom, energy_axis_true=energy_axis_true)

	maker = MapDatasetMaker()
	safe_mask_maker = SafeMaskMaker(
		methods=["aeff-default", "offset-max"], offset_max="2.5 deg"
	)

	circle = CircleSkyRegion(center=geom.center_skydir, radius=0.2 * u.deg)
	exclusion_mask = geom.region_mask([circle], inside=False)

	spectral_model = PowerLawNormSpectralModel()
	spectral_model.norm.frozen = False
	spectral_model.tilt.frozen = False
	fov_bkg_maker = FoVBackgroundMaker(
		method="fit",
		exclusion_mask=exclusion_mask,
		spectral_model=spectral_model
	)

	for obs in observations:
		dataset = maker.run(stacked, obs)
		dataset = safe_mask_maker.run(dataset, obs)
		dataset = fov_bkg_maker.run(dataset)
		stacked.stack(dataset)



It is also possible to implement other normed models, such as the
`~gammapy.modeling.models.PiecewiseNormSpectralModel`. To do so,
you can utilise most of the above code with an adaption to the ``spectral_model`` applied
in the ``fov_bkg_maker``.

.. code-block:: python

    from gammapy.modeling.models import PiecewiseNormSpectralModel

    spectral_model = PiecewiseNormSpectralModel(
        energy=energy_axis.edges,
        norms=np.ones_like(energy_axis.edges)
    )

    fov_bkg_maker = FoVBackgroundMaker(
        method="fit",
        exclusion_mask=exclusion_mask,
        spectral_model=spectral_model
    )


Note: to prevent poorly constrained ``norm`` parameters or large variance in
the last bins, the binning should be adjusted to have wider bins at higher energies.
This ensures there are enough statistics per bin, enabling the fit to converge.


.. minigallery::
    :add-heading: Examples using `~gammapy.makers.FoVBackgroundMaker`

    ../examples/tutorials/details/makers.py
    ../examples/tutorials/starting/analysis_2.py
