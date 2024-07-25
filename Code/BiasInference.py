# This script reads in tracking data from csv files, bins it, and calls the bias-persistent random walk inference for each bin
import sys
import os

sys.path.append(os.path.abspath('..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inference.walker_inference import BiasedPersistentInferer, prepare_paths
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
        # Weaver's paper spatial binning: 0-70,70-140,140-250,250-360,360-500
        s70 = trajectory[(trajectory['r'] >= 0) & (trajectory['r'] < 70)]
        s140 = trajectory[(trajectory['r'] >= 70) & (trajectory['r'] < 140)]
        s250 = trajectory[(trajectory['r'] >= 140) & (trajectory['r'] < 250)]
        s360 = trajectory[(trajectory['r'] >= 250) & (trajectory['r'] < 360)]
        s500 = trajectory[(trajectory['r'] >= 360) & (trajectory['r'] < 500)]

        return [s70,s140,s250,s360,s500]

    def space_binning2x(trajectory):
        # Weaver's paper spatial binning: 0-70,70-140,140-250,250-360,360-500
        s70 = trajectory[(trajectory['r'] >= 0) & (trajectory['r'] < 70)]
        s105 = trajectory[(trajectory['r'] >= 70) & (trajectory['r'] < 105)]
        s140 = trajectory[(trajectory['r'] >= 105) & (trajectory['r'] < 140)]
        s195 = trajectory[(trajectory['r'] >= 140) & (trajectory['r'] < 195)]
        s250 = trajectory[(trajectory['r'] >= 195) & (trajectory['r'] < 250)]
        s305 = trajectory[(trajectory['r'] >= 250) & (trajectory['r'] < 305)]
        s360 = trajectory[(trajectory['r'] >= 305) & (trajectory['r'] < 360)]
        s430 = trajectory[(trajectory['r'] >= 360) & (trajectory['r'] < 430)]
        # bin-centre distances = [54, 87.5, 122.5, 167.5, 222.5, 277.5, 332.5, 395]
        return [s70,s105,s140,s195,s250,s305,s360,s430]
    
    def time_binning(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t5  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 5)]
        t10 = space_bin[(space_bin['t'] >= 5) & (space_bin['t'] < 10)]
        t15 = space_bin[(space_bin['t'] >= 10) & (space_bin['t'] < 15)]
        t20 = space_bin[(space_bin['t'] >= 15) & (space_bin['t'] < 20)]
        t25 = space_bin[(space_bin['t'] >= 20) & (space_bin['t'] < 25 )]
        # t30 = space_bin[(space_bin['t'] >= 25) & (space_bin['t'] < 30 )]
        # t40 = space_bin[(space_bin['t'] >= 30) & (space_bin['t'] < 40 )]
        # t50 = space_bin[(space_bin['t'] >= 40) & (space_bin['t'] < 50 )]
        # t60 = space_bin[(space_bin['t'] >= 50) & (space_bin['t'] < 60.1 )]
        return [t5,t10,t15,t20,t25]

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
    time_space_bins = list(map(time_binning, distance))

    return time_space_bins

# # For half wounds need to split the data by the top half and bottom half: 
# trajectory_controlhalf, trajectory_mcrhalf = angle_binning(trajectory)

import emcee
import multiprocessing as mp

def run_inference(loadpath, loadfilename, savepath, savefilename, NWalkers, NIters, binning_function):
    trajectory = pd.read_csv(os.path.join(loadpath, loadfilename))
    trajectory = trajectory.drop(trajectory.columns[0], axis=1)

    source = PointSource(position=np.array([0, 0]))
    bin_counter = 0

    Bins = binning_function(trajectory)
    num_total_bins = len(Bins) * len(Bins[0])

    mp.set_start_method('fork', force=True)

    for i in range(len(Bins)):
        for j in range(len(Bins[0])):
            bin_counter += 1
            combined_filename = os.path.join(savepath, savefilename + '_bins{}{}'.format(i, j) + '.h5')
            if os.path.isfile(combined_filename):
                print('already analysed bin {}/{}'.format(bin_counter, num_total_bins))
            else:
                print('analysing bin {}/{}'.format(bin_counter, num_total_bins))
                backend = emcee.backends.HDFBackend(combined_filename)
                backend.reset(NWalkers, 3)
                inferer = BiasedPersistentInferer(
                    prepare_paths([paths[['x', 'y']].values for id, paths in Bins[i][j].groupby('Track_ID')],
                                  include_t=False), source)
                inferer.ensembleinfer(NWalkers, NIters, Pooling=True, savefile=backend)

# Control data
run_inference("../data/cell_tracks/Single_wound/CTR_revision", "Control_filtered_combined.csv", "../data/BP_inference/", "Single_wound_CTR_revision_nw20", 20, 10000, spatial_temporal_binning)

# MCR DATA
run_inference("../data/cell_tracks/Single_wound/MCR_revision", "MCR_filtered_combined.csv", "../data/BP_inference/", "Single_wound_MCR_revision_nw20", 20, 10000, spatial_temporal_binning)