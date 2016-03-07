"""
Example script for the new class OffDataBackgroundMaker
"""
from gammapy.background import OffDataBackgroundMaker
from gammapy.datasets import gammapy_extra

def OffDataBackgroundMaker():
    data_dir = gammapy_extra.filename('datasets/hess-crab4-hd-hap-prod2/')
    #init function of the class will have one parameter data_dir (Directory where the index files are located)
    # and we will define five members:
    #1) data_store from the data_dir
    #2) name of the run list: if None = "run.lis"
    #3) outdir: Directory where all output files go. If None = "out"
    #4) obs_table_grouped_filename: name of the obstable with the extra column of the group number. If None = self.outdir / 'obs.ecsv'
    #5) group_filename: name of the table with the different groups in zenith and efficiency. If None = self.outdir / 'group-def.ecsv'
    return OffDataBackgroundMaker(data_dir)

def background_list():
    background_maker = OffDataBackgroundMaker()
    selection= "debug"
    #Define the method background_list that create a list of OFF runs
    background_maker.background_list(selection)

def group():
    background_maker = OffDataBackgroundMaker()
    #define a class method define_obs_table() that will create self.obs_table from a OFF run_list and the obs table
    # of all the runs
    background_maker.define_obs_table(background_maker.run_list)
    #Define a method background_group that groups the previous OFF run_list in different band of zenith and efficiency
    background_maker.background_group(background_maker.outdir)

def background_model():
    background_maker = OffDataBackgroundMaker()
    #Define a method background_model that takes three params:
    #1)the modelling type= 2D ou 3D
    #2) obs_table_filename: name of the observation table with the group number for each observation.
    # If None take the self.obs_table_grouped_filename
    #3)excluded_sources: table with the radius, RADEC of the sources we want to exclude if we use the packman method in
    # the fill_obs method of the class EnergyOffsetBackgroundModel
    background_maker.background_model(modelling_type, obs_table_filename=None, excluded_sources=None)


if __name__ == '__main__':
    background_list()
    group()
    background_model()
