"""Example high-level analysis workflows.

This code doesn't run yet, it is just the result of some brainstorming how we
could structure analysis functionality in Gammapy.

The basic idea is to have "analysis" classes for the following cases:
- 0-dim region-based analysis
- 1-dim spectrum analysis
- 2-dim image analysis
- 3-dim cube analysis

Typically the analysis will be split into the following classes:

- SpectrumMaker, ImageMaker, CubeMaker
  - Computes counts and exposure and sometimes background arrays
- SpectrumObservation, ImageObservation, CubeObservation
  - Container with counts, exposure and background arrays
  - Has extra info like PSF, depending on the analysis
  - Filled by the maker classes
- SpectrumFitter, ImageFitter, CubeFitter
  - Takes lists of observation objects as input
  - Those lists could be Python lists or our own SpectrumObservationList etc,
    with convenience methods attached (like print summary, plot summary, ...)

These classes could be located in (to be discussed)

- SpectrumMaker, SpectrumObservation and SpectrumFitter in gammapy.spectrum and similarly for image and cube
- Or, have the container classes SpectrumObservation and SpectrumObservationList in spectrum
  and the "analysis" or "maker" classes in `gammapy.scripts` (which should be thought of as high-level,
  end-user pre-baked common analysis functionality, whether it's exposed via command line interface or not.

To be discussed:

- When are results attached to analysis or container classes?
  And when split out into separate "results" classes?
- OK to rename `SpectrumExtraction` to `SpectrumMaker`
  (to have uniform names with the `ImageMaker` and `CubeMaker`)?
- At the moment we use "image" and "map" to mean the same thing.
  Should we commit to one term?
"""


def get_observations():
    """Get `ObservationList`, i.e. the IACT dataset to analyse.

    This will be the common first step for all the IACT analyses.

    The idea is that `ObservationList` is the data definition / access object.
    Data is loaded from disk into memory on access.

    TODO:
    - When is data discarded from memory?
    - Should the observation grouping specification happen here,
      so that the obs grouping code isn't duplicated for the various analyses?
    """
    from gammapy.data import DataStore, ObservationList
    data_store = DataStore.from_dir(dir=...)
    obs_id = data_store.obs_table.cone_select(...)

    observations = ObservationList(data_store=..., obs_id=...)
    observations = data_store.make_obs_list(selection=...)

    return observations


def run_scalar_analysis():
    """Run 0-dim scalar analysis.

    By this we mean get scalar numbers for simple stats for a given
    region and energy band.

    Goals:
    - per-run excess, significance, livetime, ...
    - total excess, significance, livetime, ...
    """
    from gammapy.data import RegionBackgroundEstimator, BackgroundEstimate


def run_spectral_analysis():
    """Run 1-dim spectral analysis.

    Goals:
    - counts and exposure vectors
    - best-fit model values
    - joint fit on observations or stacked analysis
    - nice to have: access total stats, e.g. total significance
    - nice to have: access run-wise results
    """
    from gammapy.spectrum import SpectrumExtraction, SpectrumFit, SpectrumFitResult, SpectrumGrouping, \
        SpectrumObservation, SpectrumObservationList, SpectrumResult, SpectrumResultDict, SpectrumStats
    from gammapy.spectrum import PowerLaw
    from gammapy.data import Target, TargetSummary, DataStore
    from gammapy.data import RegionBackgroundEstimator, BackgroundEstimate

    observations = get_observations()
    target = Target(pos='dummy', on_region='dummy')
    config = dict(method='dummy', apply_psf_correction=True)

    # User can run this explicitly, can be done via spectrum_extraction.estimate_bg()
    # bg_estimator = BackgroundEstimator(observations, target, **config)
    # bg_estimator.run() # computes reflected regions and filters off lists
    #
    # # If you want, the `BackgroundEstimate` is serialisable
    # bg_estimator.background_estimate.write(dir='temp')
    # bg_estimate = BackgroundEstimate.read(dir='temp')

    spectrum_extraction = SpectrumExtraction(bg_estimate, observations, target, **config)
    spectrum_extraction.run()

    # spectrum_extraction.make_reflected_regions()
    # spectrum_extraction.bg_estimate
    # spectrum_extraction.estimate_background()
    # stats_table = spectrum_extraction.make_stats_table()

    # If you want, the `SpectrumObservationList` is serialisable
    spectrum_extraction.spectrum_observation_list.write(dir='temp')
    spectrum_obs_list = SpectrumObservationList.read(dir='temp')

    # Compute new stacked spectra with a given grouping
    group_id = 'left up to the user for now'  # Python list of group_id for each obs_id
    spectrum_obs_list = spectrum_obs_list.apply_stacking(group_id=group_id)

    model = PowerLaw()
    spectrum_fit = SpectrumFit(spectrum_obs_list, model)
    spectrum_fit.run()

    # If you want, the `SpectrumResult` is serialisable
    spectrum_fit.spectrum_result.write(dir='temp')
    spectrum_fit_result = SpectrumFitResult.read(dir='temp')


    # analysis = GtAnalysis(observations, config)
    # analysis.setup()
    # analysis.fit()


def run_image_analysis():
    """Run 2-dim image analysis.

    """
    # from gammapy.scripts import MosaicImage, ObsImage
    # ObsImage is equivalent to SpectrumObservation
    # MosaicImage is equivalent to SpectrumObservationList and SpectrumObservation
    from gammapy.image import SkyMap
    from gammapy.image import MapMaker, MapFitter, SkyMaps

    observations = get_observations()

    ref_map = SkyMap.empty('dummy')

    config = dict(method='ring')
    maker = MapMaker(observations, ref_map, **config)
    maker.run()

    maker.images

    stats_table = maker.make_stats_table()


def run_cube_analysis():
    """Run 3-dim cube analysis.

    """
    observations = get_observations()


def analyse_target_list():
    """Example how to process multiple targets.

    This example runs two analyses for each target:
    - 1-dim spectral
    - 2-dim image analy
    """
    from gammapy.spectrum import SpectrumExtraction, SpectrumFit
    from gammapy.image import ImageMaker, ImageFitter
    from gammapy.scripts import TargetList

    observations = get_observations()
    target_list = TargetList()

    for target in target_list:
        config = dict()
        analysis = SomeAnalysis(**config)
        analysis.setup(data_store, target)
        analysis.run()
        analysis.save()
