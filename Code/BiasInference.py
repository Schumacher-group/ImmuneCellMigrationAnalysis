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

def angle_binning_piby2(trajectory):
        theta_pos = trajectory[(trajectory['theta'] >= np.pi/4) & (trajectory['theta'] <= np.pi*3/4)]
        theta_neg = trajectory[(trajectory['theta'] <= - np.pi/4) & (trajectory['theta'] >= - np.pi*3/4)]

        return [theta_pos, theta_neg]

def spatial_temporal_binning(dataframe: pd.DataFrame, e2e_threshold=0):

    def space_binning(trajectory):
        # Weaver's paper spatial binning: 0-70,70-140,140-250,250-360,360-500
        s70 = trajectory[(trajectory['r'] >= 0) & (trajectory['r'] < 70)]
        s140 = trajectory[(trajectory['r'] >= 70) & (trajectory['r'] < 140)]
        s250 = trajectory[(trajectory['r'] >= 140) & (trajectory['r'] < 250)]
        s360 = trajectory[(trajectory['r'] >= 250) & (trajectory['r'] < 360)]
        s500 = trajectory[(trajectory['r'] >= 360) & (trajectory['r'] < 500)]

        return [s70,s140,s250,s360,s500]

    def space_binning_175(trajectory):
        # Weaver's paper spatial binning: 0-70,70-140,140-250,250-360,360-500
        s70 = trajectory[(trajectory['r'] >= 0) & (trajectory['r'] < 70)]
        s120 = trajectory[(trajectory['r'] >= 70) & (trajectory['r'] < 120)]
        s175 = trajectory[(trajectory['r'] >= 120) & (trajectory['r'] < 175)]
        return [s70,s120,s175]
    
    def space_binning_twowounds(trajectory):
        # Weaver's paper spatial binning: 0-70,70-140,140-250,250-360,360-500
        s57 = trajectory[(np.abs(trajectory['y']) >= 0) & (np.abs(trajectory['y']) <= 57)]
        s114 = trajectory[(np.abs(trajectory['y']) >= 57) & (np.abs(trajectory['y']) <= 114)]
        s171 = trajectory[(np.abs(trajectory['y']) >= 114) & (np.abs(trajectory['y']) <= 171)]
        return [s57,s114,s171]
    
    def space_binning_twowounds_5bins(trajectory):
        # Weaver's paper spatial binning: 0-70,70-140,140-250,250-360,360-500
        s34 = trajectory[(np.abs(trajectory['y']) >= 0) & (np.abs(trajectory['y']) <= 34)]
        s68 = trajectory[(np.abs(trajectory['y']) >= 34) & (np.abs(trajectory['y']) <= 68)]
        s102 = trajectory[(np.abs(trajectory['y']) >= 68) & (np.abs(trajectory['y']) <= 102)]
        s136 = trajectory[(np.abs(trajectory['y']) >= 102) & (np.abs(trajectory['y']) <= 136)]
        s170 = trajectory[(np.abs(trajectory['y']) >= 136) & (np.abs(trajectory['y']) <= 170)]
        return [s34,s68,s102,s136,s170]
    
    def space_binning_twowounds_5bins_edge2edge(trajectory):
        # Weaver's paper spatial binning: 0-70,70-140,140-250,250-360,360-500
        s55 = trajectory[(np.abs(trajectory['y']) >= 0) & (np.abs(trajectory['y']) <= 55)]
        s75 = trajectory[(np.abs(trajectory['y']) >= 55) & (np.abs(trajectory['y']) <= 75)]
        s95 = trajectory[(np.abs(trajectory['y']) >= 75) & (np.abs(trajectory['y']) <= 95)]
        s115 = trajectory[(np.abs(trajectory['y']) >= 95) & (np.abs(trajectory['y']) <= 115)]
        s135 = trajectory[(np.abs(trajectory['y']) >= 115) & (np.abs(trajectory['y']) <= 170)]
        return [s55,s75,s95,s115,s135]
    
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
    
    def time_binning_2bins(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t12  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 12.5)]
        t25 = space_bin[(space_bin['t'] >= 12.5) & (space_bin['t'] < 25)]
        return [t12,t25]
    
    def time_binning_2bins_20(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t10  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 10)]
        t20 = space_bin[(space_bin['t'] >= 10) & (space_bin['t'] < 20)]
        return [t10,t20]
    
    def time_binning_2bins_originalsubmission(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t10  = space_bin[(space_bin['t'] >= 7.5) & (space_bin['t'] < 12.5)]
        t20 = space_bin[(space_bin['t'] >= 17.5) & (space_bin['t'] <= 22.5)]
        return [t10,t20]
    
    def time_binning_2bins_5min(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t10  = space_bin[(space_bin['t'] >= 5.0) & (space_bin['t'] < 10.0)]
        t20 = space_bin[(space_bin['t'] >= 15.0) & (space_bin['t'] <= 20.0)]
        return [t10,t20]
    
    def time_binning_2bins_3min(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t10  = space_bin[(space_bin['t'] >= 7.0) & (space_bin['t'] < 10.0)]
        t20 = space_bin[(space_bin['t'] >= 17.0) & (space_bin['t'] <= 20.0)]
        return [t10,t20]
    
    def time_binning_2bins_2min(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t10  = space_bin[(space_bin['t'] >= 8.0) & (space_bin['t'] < 10.0)]
        t20 = space_bin[(space_bin['t'] >= 18.0) & (space_bin['t'] <= 20.0)]
        return [t10,t20]
    
    def time_binning_3bins_20(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t10  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 10)]
        t18 = space_bin[(space_bin['t'] >= 10) & (space_bin['t'] < 18)]
        t20 = space_bin[(space_bin['t'] >= 18) & (space_bin['t'] <= 20)]
        return [t10,t18,t20]
    
    def time_binning_4bins_twowounds(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t10  = space_bin[(space_bin['t'] >= 5) & (space_bin['t'] <= 15)]
        t20 = space_bin[(space_bin['t'] >= 15) & (space_bin['t'] <= 25)]
        t30 = space_bin[(space_bin['t'] >= 25) & (space_bin['t'] <= 35)]
        t40 = space_bin[(space_bin['t'] >= 35) & (space_bin['t'] <= 45)]
        return [t10,t20,t30,t40]
    
    def time_binning_WeaversFirst2bins(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t20  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 20)]
        t25 = space_bin[(space_bin['t'] >= 20) & (space_bin['t'] < 25)]
        return [t20,t25]
    
    def time_binning_WeaversFirst2bins_excl5(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t20  = space_bin[(space_bin['t'] >= 5) & (space_bin['t'] < 20)]
        t25 = space_bin[(space_bin['t'] >= 20) & (space_bin['t'] < 25)]
        return [t20,t25]
    
    def time_binning_WeaversFirst1bin(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t20  = space_bin[(space_bin['t'] >= 0) & (space_bin['t'] < 20)]
        return [t20]
    
    def time_binning_WeaversFirst1bins_excl5(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t25  = space_bin[(space_bin['t'] >= 5) & (space_bin['t'] < 25)]
        return [t25]
    
    def time_binning_WeaversFirst1bins_excl10(space_bin):
        # Weavers paper temporal binning:  0-20, 20-35, 35-50, 50-65, 65-90, 90-125
        t25  = space_bin[(space_bin['t'] >= 10) & (space_bin['t'] < 25)]
        return [t25]

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
    
    def exclude_end_to_end_distances(trajectories, e2e_threshold):
        # exclude trajectories below a specified end to end distance
        # make a list of the unique trackIDs
        track_ids = trajectories.Track_ID.unique()
        # loop through track_ids and remove the ones with e2e distances below the threshold
        for track_id in track_ids:
            # calculate the e2e distance for the track
            this_trajectory = trajectories[trajectories.Track_ID == track_id]
            this_e2e_distance = np.linalg.norm(this_trajectory[['x', 'y']].iloc[0] - this_trajectory[['x', 'y']].iloc[-1])
            if this_e2e_distance < e2e_threshold:
                # drop track_id from trajectories dataframe
                trajectories = trajectories[trajectories.Track_ID != track_id]
        
        return trajectories

    if e2e_threshold>0:
        trajectories = exclude_end_to_end_distances(dataframe, e2e_threshold)
    elif e2e_threshold==0:
        trajectories = dataframe

    distance_bins = space_binning_twowounds_5bins(trajectories)
    
    time_space_bins = list(map(time_binning_4bins_twowounds, distance_bins))

    return time_space_bins

# # For half wounds need to split the data by the top half and bottom half: 
# trajectory_controlhalf, trajectory_mcrhalf = angle_binning(trajectory)

import emcee
import multiprocessing as mp

filesuffix = "_s5y"
halfwound_toggle = False

def run_inference(loadpath, loadfilename, savepath, savefilename, NWalkers, NIters, binning_function, 
                  e2e_threshold=0, farhalf=False, angle_bin=0, halfwound=False):
    trajectory = pd.read_csv(os.path.join(loadpath, loadfilename))
    trajectory = trajectory.drop(trajectory.columns[0], axis=1)

    source = PointSource(position=np.array([0, 0]))
    bin_counter = 0

    if farhalf:
        print("Using only the far half of the wound...")
        halfwound = True
        angle_bin = 0
    # bin tracks into angles for half-wound experiments
    if halfwound==True:
        if (angle_bin>=0) & (angle_bin<=1):
            print("Binning tracks into each side of the wound...")
            top_half, bottom_half = angle_binning(trajectory)
            if angle_bin==0:
                print("Analyzing tracks on the control half of the tissue...")
                trajectory = top_half
            elif angle_bin==1:
                print("Analyzing tracks on the MCR KD half of the tissue...")
                trajectory = bottom_half
        elif (angle_bin>=2) & (angle_bin<=3):
            print("Binning tracks into pi/2 on each side of the wound...")
            top_quarter, bottom_quarter = angle_binning_piby2(trajectory)
            if angle_bin==2:
                print("Analyzing tracks on the control half of the tissue...")
                trajectory = top_quarter
            elif angle_bin==3:
                print("Analyzing tracks on the MCR KD half of the tissue...")
                trajectory = bottom_quarter

    Bins = binning_function(trajectory, e2e_threshold=e2e_threshold)
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

# # Control data
# run_inference("../data/cell_tracks/Single_wound/CTR_revision", "Control_filtered_combined.csv", 
#               "../data/BP_inference/", "Single_wound_CTR_revision"+filesuffix, 10, 10000, spatial_temporal_binning, 
#               e2e_threshold=0, farhalf=True)

# # MCR DATA
# run_inference("../data/cell_tracks/Single_wound/MCR_revision", "MCR_filtered_combined.csv",
#                "../data/BP_inference/", "Single_wound_MCR_revision"+filesuffix, 10, 10000, spatial_temporal_binning,
#                 e2e_threshold=0, farhalf=True)

# halfwound experiments
# # Control data
# run_inference("../data/cell_tracks/Half_wound/CTR_revision", "Halfwound_CTR_filtered_combined.csv", 
#               "../data/BP_inference/", "Halfwound_CTR_revision"+filesuffix, 10, 10000, spatial_temporal_binning,
#               halfwound=False)

# # MCR DATA - control half
# run_inference("../data/cell_tracks/Half_wound/MCR_revision", "Halfwound_MCR_filtered_combined.csv",
#                "../data/BP_inference/", "Halfwound_MCR_revision_ctr_half"+filesuffix, 10, 10000, spatial_temporal_binning,
#                 halfwound=True, angle_bin=0)

# # MCR DATA - MCR half
# run_inference("../data/cell_tracks/Half_wound/MCR_revision", "Halfwound_MCR_filtered_combined.csv",
#                "../data/BP_inference/", "Halfwound_MCR_revision_mcr_half"+filesuffix, 10, 10000, spatial_temporal_binning,
#                 halfwound=True, angle_bin=1)

# # MCR DATA - control quarter
# run_inference("../data/cell_tracks/Half_wound/MCR_revision", "Halfwound_MCR_filtered_combined.csv",
#                "../data/BP_inference/", "Halfwound_MCR_revision_ctr_quarter"+filesuffix, 10, 10000, spatial_temporal_binning,
#                 halfwound=True, angle_bin=2)

# # MCR DATA - MCR quarter
# run_inference("../data/cell_tracks/Half_wound/MCR_revision", "Halfwound_MCR_filtered_combined.csv",
#                "../data/BP_inference/", "Halfwound_MCR_revision_mcr_quarter"+filesuffix, 10, 10000, spatial_temporal_binning,
#                 halfwound=True, angle_bin=3)

# halfwound experiments - two wound experiments
# # Control data - top half
run_inference("../data/cell_tracks/Two_wound_data/CTR_revision", "Two_wound_Control_top_filtered_combined.csv",
               "../data/BP_inference/", "Two_wound_CTR_revision_top"+filesuffix, 10, 10000, spatial_temporal_binning,
               halfwound=halfwound_toggle, angle_bin=3)

# # Control data - bottom half
run_inference("../data/cell_tracks/Two_wound_data/CTR_revision", "Two_wound_Control_bottom_filtered_combined.csv",
               "../data/BP_inference/", "Two_wound_CTR_revision_bottom"+filesuffix, 10, 10000, spatial_temporal_binning,
                halfwound=halfwound_toggle, angle_bin=2)

# # MCR DATA - top half
run_inference("../data/cell_tracks/Two_wound_data/MCR_revision", "Two_wound_MCR_top_filtered_combined.csv",
               "../data/BP_inference/", "Two_wound_MCR_revision_top"+filesuffix, 10, 10000, spatial_temporal_binning,
               halfwound=halfwound_toggle, angle_bin=3)

# # MCR DATA - bottom half
run_inference("../data/cell_tracks/Two_wound_data/MCR_revision", "Two_wound_MCR_bottom_filtered_combined.csv",
               "../data/BP_inference/", "Two_wound_MCR_revision_bottom"+filesuffix, 10, 10000, spatial_temporal_binning,
                halfwound=halfwound_toggle, angle_bin=2)