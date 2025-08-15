# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Gammapy documentation ordering configuration file.
#
# GalleryExplicitOrder is a re-implementation of private sphinx_gallery class `_SortKey`.
#


TUTORIAL_SORT_DICT = {
    #  **Tutorials**
    #  introductory
        "overview.py": 0,
        "analysis_1.py": 1,
        "analysis_2.py": 2,
    #  data exploration
        "hess.py": 0,
        "cta.py": 1,
        "magic.py": 2,
        "fermi_lat.py": 3,
        "hawc.py": 4,
        "veritas.py": 5,
    # details
        "irfs.py": 0,
        "observation_clustering.py": 1,
        "theta_square_plot.py": 2,
        "makers.py": 3,
        "datasets.py": 4,
        "maps.py": 5,
        "mask_maps.py": 6,
        "catalog.py": 7,
        "models.py": 8,
        "priors.py": 9,
        "model_management.py": 10,
        "fitting.py": 11,
        "nested_sampling_Crab.py": 12,
        "estimators.py": 13,
        "parameter_ul.py": 14,
    #  data analysis
    #  1d
        "cta_sensitivity.py": 0,
        "spectral_analysis.py": 1,
        "spectral_analysis_hli.py": 2,
        "extended_source_spectral_analysis.py": 3,
        "spectrum_simulation.py": 4,
        "sed_fitting.py": 5,
    #  2d
        "detect.py": 0,
        "ring_background.py": 1,
        "modeling_2D.py": 2,
    # 3d
        "analysis_3d.py": 0,
        "cta_data_analysis.py": 1,
        "analysis_mwl.py": 2,
        "simulate_3d.py": 3,
        "event_sampling.py": 4,
        "event_sampling_nrg_depend_models.py": 5,
        "flux_profiles.py": 6,
        "non_detected_source.py": 7,
    # time
        "light_curve.py": 0,
        "light_curve_flare.py": 1,
        "variability_estimation.py": 2,
        "time_resolved_spectroscopy.py": 3,
        "light_curve_simulation.py": 4,
    # astro
        "ebl.py": 0,
        "energy_dependent_estimation.py": 1,
        "pulsar_analysis.py": 2,
        "astro_dark_matter.py": 3,
    # scripts
        "survey_map.py": 0,
}


class BaseExplicitOrder:
    """
    Base class inspired by sphinx_gallery _SortKey to customize sorting based on a
    dictionary. The dictionary should contain the filename and its order in the
    subsection.
    """

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __init__(self, src_dir):
        self.src_dir = src_dir


class TutorialExplicitOrder(BaseExplicitOrder):
    """
    Class that handle the ordering of the tutorials in each gallery subsection.
    """

    sort_dict = TUTORIAL_SORT_DICT

    def __call__(self, filename):
        if filename in self.sort_dict.keys():
            return self.sort_dict.get(filename)
        else:
            return 0
