"""
This script converts the trajectories.csv file from the rdata package to a workable pandas dataframe for our inference
pipeline.
"""
import sys
import os

sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
from inference.walker_inference import BiasedPersistentInferer, prepare_paths
from in_silico.sources import PointSource

# This takes our csv's and loads them into a dataframe
df1 = pd.read_csv("../data/Comparison_data/140514Wounded/trajectories1.csv")
df2 = pd.read_csv("../data/Comparison_data/280514WoundedS20/trajectories1.csv")
df3 = pd.read_csv("../data/Comparison_data/280514WoundedS22/trajectories1.csv")
"""
create_dataframe function takes in the loaded csv file as a new dataframe and formats it to work within our inference 
pipeline. We first create a new column labelled ("Track_ID") so each of our tracks can individually be identified, and 
then create a new label value for individual tracks, within the for loop. We then convert our time column into int 
from string and convert it into seconds. Finally, we reshape the data to fit the inference pipeline and remove any
superfluous columns (r,z).
"""



def create_dataframe(df, xw, yw,loc):
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
    reshapeddata = pd.DataFrame({'Track_ID': df['Track_ID'], 't': df['t'], 'x': df['X'], 'y': df['Y']})
    reshapeddata['x'] = reshapeddata['x'] - xw
    reshapeddata['y'] = reshapeddata['y'] - yw
    reshapeddata['r'] = (lambda x, y: np.sqrt(x ** 2 + y ** 2))(reshapeddata['x'], reshapeddata['y'])
    reshapeddata['Track_ID'] = reshapeddata['Track_ID'].astype(str)
    reshapeddata.Track_ID = reshapeddata.Track_ID + "{}".format(loc)  # creates a label for the tracks to be organised by
    return reshapeddata




# Reshaped dataframes to work in our pipeline
wound_x = 300
wound_y = 500
trajectory_1 = create_dataframe(df1, wound_x, wound_y, "A")
wound_x = 160
wound_y = 270
trajectory_2 = create_dataframe(df2, wound_x, wound_y, "B")
wound_x = 160
wound_y = 270
trajectory_3 = create_dataframe(df3, wound_x, wound_y, "C")

FramesCont = [trajectory_1, trajectory_2, trajectory_3] # exclude dataframes 3 and 5, they are the control for the mutant tissue
trajectory = pd.concat(FramesCont) #Concatenates the control data

def space_slice(theta):
    s70 = theta[(theta['r'] >= 0) & (theta['r'] <= 70)]
    s140 = theta[(theta['r'] >= 70) & (theta['r'] <= 140)]
    s250 = theta[(theta['r'] >= 140) & (theta['r'] <= 250)]
    s360 = theta[(theta['r'] >= 250) & (theta['r'] <= 360)]
    s500 = theta[(theta['r'] >= 360) & (theta['r'] <= 500)]
    return [s70, s140, s250, s360, s500]


def time_slice(space):
    t20 = space[(space['t'] >= 0) & (space['t'] <= (20 * 60))]
    t35 = space[(space['t'] >= (20 * 60)) & (space['t'] <= (35 * 60))]
    t50 = space[(space['t'] >= (35 * 60)) & (space['t'] <= (50 * 60))]
    t65 = space[(space['t'] >= (50 * 60)) & (space['t'] <= (65 * 60))]
    t90 = space[(space['t'] >= (65 * 60)) & (space['t'] <= (90 * 60))]
    t125 = space[(space['t'] >= (90 * 60)) & (space['t'] <= (125 * 60))]

    return [t20, t35, t50, t65, t90, t125]


distance = space_slice(trajectory)
s_distance = []
for i in range(len(distance)):
    s_distance.append(time_slice(distance[i]))

# This will run the inference method iteratively for each temporal and spatial bin and save
# them as a numpy array for analysis in the data analysis Python script

# This is important and needs to be changed to include an error note if PointWound is used instead of PointSource

source = PointSource(position=np.array([0, 0]))
NWalkers = 100
NIters = 1000
t = 0
timer = (len(distance) * len(s_distance[0]))
for i in range(len(distance)):
    for j in range(len(s_distance[0])):
        t += 1  # Tracks the number of bins
        print('analysing bin {}/{}'.format(t,timer))  # to give an overall sense of progress
        inferer = BiasedPersistentInferer(
            prepare_paths([paths[['x', 'y']].values for id, paths in s_distance[i][j].groupby('Track_ID')],
                          include_t=False), source)
        inf_out = inferer.ensembleinfer(NWalkers, NIters)
        np.save(
            '/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/WalkerData/WeaversData/WeaversL1{}{}'.format(
                i, j), inf_out)
