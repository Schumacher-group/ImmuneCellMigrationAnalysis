# Import all the necessary modules needed to run the inference pipeline
import sys
import os

sys.path.append(os.path.abspath('..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from inference.walker_inference import BiasedPersistentInferer, prepare_paths, spatial_temporal_binning
from inference.attractant_inference import AttractantInferer
from in_silico.sources import PointSource, PointWound
#from Utilities.plotting import plot_find_wound_location,plotxy_space_bins,plotxy_time_bins,observed_bias_plotting

# This takes our csv's and loads them into a dataframe
#df = pd.read_csv('/Users/danieltudor/Documents/Groups/Wood group/New Images from Luigi/Control_2_cleaned_noFB')
#df2 = pd.read_csv('/Users/danieltudor/Documents/Groups/Wood group/New Images from Luigi/Control_3_cleaned')
#df3 = pd.read_csv('/Users/danieltudor/Documents/Groups/Wood group/New Images from Luigi/Control_5_cleaned')


def create_dataframe(df, xw, yw,loc):
    """
    df["Track_ID"] = 0
    row_labels = df[df.X == "x"].index.values
    row_labels = np.append(row_labels, len(df))
    for row in range(len(row_labels) - 1):
        df.iloc[row_labels[row]:row_labels[row + 1], df.columns.get_loc('Track_ID')] = row + 1
    df = df.drop(df.index[row_labels - 1])
    df['t'] = pd.to_numeric(df['t'])
    df['X'] = pd.to_numeric(df['X'])
    df['Y'] = pd.to_numeric(df['Y'])
    df["t"] = 60 * df["t"]   
    
    
    """

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

def spatial_temporal_binning(dataframe: pd.DataFrame):
    def space_binning(trajectory):
        # Weaver's paper spatial binning: 0-35,35-70,70-140,140-250,250-360,360-500
        s30 = trajectory[(trajectory['r'] >= 0) & (trajectory['r'] < 40)]
        s60 = trajectory[(trajectory['r'] >= 40) & (trajectory['r'] < 80)]
        s90 = trajectory[(trajectory['r'] >= 80) & (trajectory['r'] < 120)]
        s130 = trajectory[(trajectory['r'] >= 120) & (trajectory['r'] < 160)]
        s150 = trajectory[(trajectory['r'] >= 160) & (trajectory['r'] < 200)]
        s180 = trajectory[(trajectory['r'] >= 200) & (trajectory['r'] < 240)]
        s210 = trajectory[(trajectory['r'] >= 240) & (trajectory['r'] < 280)]
        #s250 = trajectory[(trajectory['r'] >= 210) & (trajectory['r'] < 250)]

        return [s30,s60, s90,s130,s150,s180,s210]#, s250]

    def time_binning(space_bin):
        # Weavers paper temporal binning: 0-5,5-10, 20-35,35-50,50-60    ,65-90,90-125
        t5  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 5)]
        t10 = space_bin[(space_bin['t'] >= 5) & (space_bin['t'] < 10)]
        t15 = space_bin[(space_bin['t'] >= 10) & (space_bin['t'] < 15)]
        t20 = space_bin[(space_bin['t'] >= 15) & (space_bin['t'] < 20)]
        t25 = space_bin[(space_bin['t'] >= 20) & (space_bin['t'] < 25 )]
        t30 = space_bin[(space_bin['t'] >= 25) & (space_bin['t'] < 30 )]
        t40 = space_bin[(space_bin['t'] >= 30) & (space_bin['t'] < 40 )]
        t50 = space_bin[(space_bin['t'] >= 40) & (space_bin['t'] < 50 )]
        t60 = space_bin[(space_bin['t'] >= 50) & (space_bin['t'] < 61 )]
        return [t5,t10,t15,t20,t25,t30,t40,t50,t60]

    def Over_lapping_time_binning(space_bin):
        # Weavers paper temporal binning: 0-5,5-10, 20-35,35-50,50-60    ,65-90,90-125
        t5  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 5)]
        t10 = space_bin[(space_bin['t'] >= 2.5) & (space_bin['t'] < 7.5)]
        t15 = space_bin[(space_bin['t'] >= 5) & (space_bin['t'] < 10)]
        t20 = space_bin[(space_bin['t'] >= 7.5) & (space_bin['t'] < 12.5)]
        t25 = space_bin[(space_bin['t'] >= 10) & (space_bin['t'] < 15 )]
        t30 = space_bin[(space_bin['t'] >= 12.5) & (space_bin['t'] < 17.5 )]
        t40 = space_bin[(space_bin['t'] >= 15) & (space_bin['t'] < 20 )]
        #t50 = space_bin[(space_bin['t'] >= 40) & (space_bin['t'] < 50 )]
        #t60 = space_bin[(space_bin['t'] >= 50) & (space_bin['t'] < 61 )]
    

  # Bins for comparison against 15 mins wound = (0-15)(15-30)(30-45)(45-60)(60-75)(75-90)
        return [t5,t10,t15,t20,t25,t30,t40]
    distance = space_binning(dataframe)
    time_space_bins = list(map(Over_lapping_time_binning, distance))

    return time_space_bins
"""
trajectory = create_dataframe(df,235,(565.688-322),"A")
trajectory_2 = create_dataframe(df2, 246,(565.688-322),"B")
trajectory_3 = create_dataframe(df3, 235,(565.688-324),"C")


trajectory['t'] = trajectory['t'].round()
trajectory_2['t'] = trajectory_2['t'].round()
trajectory_3['t'] = trajectory_3['t'].round()
# Convert to minutes to make binning easier 
trajectory['t'] = trajectory['t'].div(60)
trajectory_2['t'] = trajectory_2['t'].div(60)
trajectory_3['t'] = trajectory_3['t'].div(60)
trajectory['t'],trajectory_2['t'],trajectory_3['t']
FramesCont = [trajectory,trajectory_2, trajectory_3] # exclude dataframes 3 and 5, they are the control for the mutant tissue
Final_trajectory = pd.concat(FramesCont) #Concatenates the control data
"""
trajectory = pd.read_csv("/Users/danieltudor/Documents/Groups/Wood group/New Images from Luigi/Control csv data/Control_filtered_spots_all_new")
trajectory = trajectory.drop(trajectory.columns[0], axis = 1 )

Bins = spatial_temporal_binning(trajectory)

# Inference pipeline for BP_Walker 
source = PointSource(position=np.array([0,0])) # Source is a position at 0,0 due to readjustment of tracks earlier
NWalkers = 80 # 100 walkers and 1000 iterations seem to give the best convergence of the emcee 
NIters = 2000
t = 0
total_bins = (len(Bins) * len(Bins[0])) # total number of bins to run the inference on 

for i in range(len(Bins)):
    for j in range(len(Bins[0])):
        t += 1  # Tracks the number of bins
        print('analysing bin {}/{}'.format(t,total_bins))  # to give an overall sense of progress
        inferer = BiasedPersistentInferer(
            prepare_paths([paths[['x', 'y']].values for id, paths in Bins[i][j].groupby('Track_ID')],
                          include_t=False), source) # prepares the data for running the inference 
        inf_out = inferer.ensembleinfer(NWalkers, NIters, Pooling = False) # calls the emcee inferer 
        np.save(
            '../data/New_control_data/control_data{}{}_timebins_trajs_new-binning-overlapping'.format(
                i, j), inf_out) # Saves to local data file 