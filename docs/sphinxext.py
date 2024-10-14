# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Gammapy documentation ordering configuration file.
#
# GalleryExplicitOrder is a re-implemation of private sphinx_gallery class `_SortKey`.
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
        "fermi_lat.py": 2,
        "hawc.py": 3,
    #  data analysis
    #  1d
        "cta_sensitivity.py": 0,
        "spectral_analysis.py": 1,
        "spectral_analysis_hli.py": 2,
        "spectral_analysis_rad_max.py": 3,
        "extended_source_spectral_analysis.py": 4,
        "spectrum_simulation.py": 5,
        "sed_fitting.py": 6,
        "ebl.py": 7,
    #  2d
        "detect.py": 0,
        "ring_background.py": 1,
        "modeling_2D.py": 2,
    # 3d
        "analysis_3d.py": 0,
        "cta_data_analysis.py": 1,
        "energy_dependent_estimation.py": 2,
        "analysis_mwl.py": 3,
        "simulate_3d.py": 4,
        "event_sampling.py": 5,
        "event_sampling_nrg_depend_models.py": 6,
        "flux_profiles.py": 7,
    # time
        "light_curve.py": 0,
        "light_curve_flare.py": 1,
        "variability_estimation.py": 2,
        "time_resolved_spectroscopy.py": 3,
        "light_curve_simulation.py": 4,
        "pulsar_analysis.py": 5,
    # api
        "observation_clustering.py": 9,
        "irfs.py": 0,
        "maps.py": 4,
        "mask_maps.py": 5,
        "makers.py": 7,
        "datasets.py": 6,
        "models.py": 1,
        "priors.py": 2,
        "model_management.py": 10,
        "fitting.py": 11,
        "estimators.py": 12,
        "catalog.py": 8,
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
