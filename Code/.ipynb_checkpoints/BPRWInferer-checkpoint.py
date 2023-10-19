import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
from inference.walker_inference import BiasedPersistentInferer, prepare_paths
from in_silico.sources import PointSource


# Variables needed for Ensemble Monte Carlo
n_iter = 5000
n_walkers = 80


"""
space_slice splits the dataframe into different spatial bins. time_slice splits the data into different temporal bins,
this allows for the inference pipeline to calculate the spatial-temporal values for the bias, persistence, and weights. 
"""
def space_slice(theta):
    s25 = theta[(theta['r'] >= 5)  & (theta['r'] <= 45)]
    s50 = theta[(theta['r'] >= 25)  & (theta['r'] <= 75)]
    s75 = theta[(theta['r'] >= 45)  & (theta['r'] <= 105)]
    s100 = theta[(theta['r'] >= 65)  & (theta['r'] <= 135)]
    s125 = theta[(theta['r'] >= 85)  & (theta['r'] <= 165)]
    s150 = theta[(theta['r'] >= 105)  & (theta['r'] <= 195)]
    s175 = theta[(theta['r'] >= 125)  & (theta['r'] <= 225)]
    return [s25,s50,s75,s100,s125,s150,s175]


def time_slice(space):
    t5 = space[(space['t'] >= 0) & (space['t'] <= 600)]
    t15 = space[(space['t'] >= 180) & (space['t'] <= 1620)]
    t30 = space[(space['t'] >= 900) & (space['t'] <= 2700)]
    t50 = space[(space['t'] >= 2100) & (space['t'] <= 3900)]
    times = [t5, t15, t30, t50]
    return times




"""
First, we slice the data into their respective spatial bins and then we split the spatial bins further to bin them in 
their respective temporal bins. When then run the inference pipeline on the space_time_dataframe. 
"""
trajectory = np.load('../data/Trajectory_dataframes/2021_06_11-04:06:07_PM_control',allow_pickle=True)
distance = space_slice(trajectory)
space_time_dataframe = [time_slice(distance[i]) for i in range(len(distance))]

"""
This will run the inference method iteratively for each temporal and spatial bin and save
them as a numpy array for analysis in the data analysis Python script.
"""
# Error term to make sure we catch any issues where PointSource isn't used as the source


source = PointSource(position=np.array([0, 0]))
"""
if source != PointSource:
    print("Cannot run bias persistence inference pipeline, please check that source is PointSource")
else:
"""
k = 0
time = space_time_dataframe[0]

for i in range(len(space_time_dataframe)):
    for j in range(len(time)):
        k += 1  # Tracks the number of bins
        print('analysing bin {}/{}'.format(k, (len(distance) * len(time))))  # to give an overall sense of progress
        inferer = BiasedPersistentInferer(
            prepare_paths(
                [paths[['x', 'y']].values for id, paths in space_time_dataframe[i][j].groupby('trackID')],
                include_t=False), source)
        inf_out = inferer.ensembleinfer(n_walkers, n_iter)
        np.save(f'../data/WalkerData/PosterData/Control_wound_{i}{j}_SMB', inf_out)
