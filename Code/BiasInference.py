# This script reads in tracking data from csv files, bins it, and calls the bias-persistent random walk inference for each bin
import sys
import os

sys.path.append(os.path.abspath('..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inference.walker_inference import BiasedPersistentInferer, prepare_paths, spatial_temporal_binning
from in_silico.sources import PointSource


def create_dataframe(df, xw, yw,loc):
    reshapeddata = pd.DataFrame({'Track_ID':df['TRACK_ID'],
                             't':df['POSITION_T'],
                             'x':df['POSITION_X'],
                             'y':df['POSITION_Y']})    
    reshapeddata['x'] = reshapeddata['x'] - xw
    reshapeddata['y'] = reshapeddata['y'] - yw
    reshapeddata['r'] = (lambda x, y: np.sqrt(x ** 2 + y ** 2))(reshapeddata['x'], reshapeddata['y'])
    reshapeddata['Track_ID'] = reshapeddata['Track_ID'].astype(str)
    reshapeddata.Track_ID = reshapeddata.Track_ID + "{}".format(loc)  # creates a label for the tracks to be organised by
    return reshapeddata

def angle_binning(trajectory):
        theta_pos = trajectory[(trajectory['theta'] >= 0)]
        theta_neg = trajectory[(trajectory['theta'] < 0)]

        return [theta_pos, theta_neg]

def spatial_temporal_binning(dataframe: pd.DataFrame, angle = False ):


    def space_binning(trajectory):
        # Weaver's paper spatial binning: 0-35,35-70,70-140,140-250,250-360,360-500
        s40 = trajectory[(trajectory['r'] >= 0) & (trajectory['r'] < 40)]
        s80 = trajectory[(trajectory['r'] >= 40) & (trajectory['r'] < 80)]
        s120 = trajectory[(trajectory['r'] >= 80) & (trajectory['r'] < 120)]
        s160 = trajectory[(trajectory['r'] >= 120) & (trajectory['r'] < 160)]
        s200 = trajectory[(trajectory['r'] >= 160) & (trajectory['r'] < 200)]
        s240 = trajectory[(trajectory['r'] >= 200) & (trajectory['r'] < 240)]
        s280 = trajectory[(trajectory['r'] >= 240) & (trajectory['r'] < 280)]

        return [s40,s80, s120,s160,s200,s240,s280]

    def time_binning(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t5  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 5)]
        t10 = space_bin[(space_bin['t'] >= 5) & (space_bin['t'] < 10)]
        t15 = space_bin[(space_bin['t'] >= 10) & (space_bin['t'] < 15)]
        t20 = space_bin[(space_bin['t'] >= 15) & (space_bin['t'] < 20)]
        t25 = space_bin[(space_bin['t'] >= 20) & (space_bin['t'] < 25 )]
        t30 = space_bin[(space_bin['t'] >= 25) & (space_bin['t'] < 30 )]
        t40 = space_bin[(space_bin['t'] >= 30) & (space_bin['t'] < 40 )]
        t50 = space_bin[(space_bin['t'] >= 40) & (space_bin['t'] < 50 )]
        t60 = space_bin[(space_bin['t'] >= 50) & (space_bin['t'] < 60.1 )]
        return [t5,t10,t15,t20,t25,t30,t40,t50,t60]

    def Over_lapping_time_binning(space_bin):
        t2  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 5)]
        t5 = space_bin[(space_bin['t'] >= 2.5) & (space_bin['t'] < 7.5)]
        t7 = space_bin[(space_bin['t'] >= 5) & (space_bin['t'] < 10)]
        t10 = space_bin[(space_bin['t'] >= 7.5) & (space_bin['t'] < 12.5)]
        t12= space_bin[(space_bin['t'] >= 10) & (space_bin['t'] < 15 )]
        t15 = space_bin[(space_bin['t'] >= 12.5) & (space_bin['t'] < 17.5 )]
        t17 = space_bin[(space_bin['t'] >= 15) & (space_bin['t'] < 20 )]
        t20 = space_bin[(space_bin['t'] >= 17.5) & (space_bin['t'] < 22.5)]
        t22 = space_bin[(space_bin['t'] >= 20.0) & (space_bin['t'] < 25.0)]
        return [t2,t5,t7,t10,t12,t15,t17,t20,t22]


    distance = space_binning(dataframe)
    time_space_bins = list(map(Over_lapping_time_binning, distance))

    return time_space_bins

loadpath = "../data/cell_tracks/Single_wound/CTR_revision/"
loadfilename = "Control_filtered_combined"
savepath = "../data/BP_inference/"
savefilename = "Single_wound_CTR_revision"
trajectory = pd.read_csv(loadpath+loadfilename)
# When we exported the CSV, it exported the index values as the first column so we need to get rid of the extra column
trajectory = trajectory.drop(trajectory.columns[0], axis = 1 ) 

# # For half wounds Lets split the data by the top half and bottom half: 
# trajectory_controlhalf, trajectory_mcrhalf = angle_binning(trajectory)

# Inference pipeline for BP_Walker 
source = PointSource(position=np.array([0,0])) # Source is a position at 0,0 due to readjustment of tracks earlier
NWalkers = 30 
NIters = 5000
bin_counter = 0

Bins = spatial_temporal_binning(trajectory)

total_bins = (len(Bins) * len(Bins[0])) # total number of bins to run the inference on 

import emcee
import multiprocessing as mp
mp.set_start_method('fork', force=True)

for i in range(len(Bins)):
    for j in range(len(Bins[0])):
        bin_counter += 1  # Tracks the number of bins
        print('analysing bin {}/{}'.format(bin_counter,total_bins))  # to give an overall sense of progress
        backend = emcee.backends.HDFBackend(savepath+savefilename+'_bins{}{}'.format(i, j)+'.h5')
        backend.reset(NWalkers, 3)
        inferer = BiasedPersistentInferer(
            prepare_paths([paths[['x', 'y']].values for id, paths in Bins[i][j].groupby('Track_ID')],
                          include_t=False), source) # prepares the data for running the inference 
        inf_out = inferer.ensembleinfer(NWalkers, NIters, Pooling = True, savefile=backend) # calls the emcee inferer 
        # np.save(
        #     savepath+savefilename+'_bins{}{}'.format(i, j), inf_out) # Saves to local data file 