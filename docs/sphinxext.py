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
        "ebl.py": 5,
        "sed_fitting.py": 6,
        "spectrum_simulation.py": 7,
    #  2d
        "detect.py": 0,
        "ring_background.py": 1,
        "modeling_2D.py": 2,
    # 3d
        "cta_data_analysis.py": 0,
        "analysis_3d.py": 1,
        "flux_profiles.py": 2,
        "energy_dependent_estimation.py": 3,
        "analysis_mwl.py": 4,
        "simulate_3d.py": 5,
        "event_sampling.py": 6,
        "event_sampling_nrg_depend_models.py": 7,
    # time
        "light_curve.py": 0,
        "light_curve_flare.py": 1,
        "variability_estimation.py": 2,
        "time_resolved_spectroscopy.py": 3,
        "pulsar_analysis.py": 4,
        "light_curve_simulation.py": 5,
    # api
        "irfs.py": 0,
        "models.py": 1,
        "priors.py": 2,
        "astro_dark_matter.py": 3,
        "maps.py": 4,
        "mask_maps.py": 5,
        "datasets.py": 6,
        "makers.py": 7,
        "catalog.py": 8,
        "observation_clustering.py": 9,
        "model_management.py": 10,
        "fitting.py": 11,
        "estimators.py": 12,
    # scripts
        "survey_map.py": 0,
}


class BaseExplicitOrder:
    """
    Base class inspired by sphinx_gallery _SortKey to customize sorting based on a
    dictionnary. The dictionnary should contain the filename and its order in the
    subsection.
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, filename):
        if filename in self.sort_dict.keys():
            return self.sort_dict.get(filename)
        else:
            return 0


class TutorialExplicitOrder(BaseExplicitOrder):
    """
    Class that handle the ordering of the tutorials in each gallery subsection.
    """

    sort_dict = TUTORIAL_SORT_DICT


