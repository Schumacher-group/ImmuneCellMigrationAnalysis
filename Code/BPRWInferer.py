import os
import sys
import numpy as np
import pandas as pd
from inference.walker_inference import BiasedPersistentInferer, prepare_paths
from in_silico.sources import PointSource

sys.path.append(os.path.abspath('..'))

# Variables needed for Ensemble Monte Carlo
n_iter = 1000
n_walkers = 50

"""
Converts the csv file into a panda dataframe and updates to include the distance from the
wound edge for spatial slicing
"""

# C and E files are internal controls
"""
The loadfile function takes the file location in the correct directory, and the wound (x,y) location. It then runs the
extract function to convert the dataframe into the correct format for processing the bias persistence random walk
inference pipeline. The output is a dataframe which has a normalised cell trajectories. 
"""


def loadfile(name, loc, x, y):
    file = f'../data/ImageJcsvs/Control_{name}_new.csv'
    file = pd.read_csv(file, header=0)
    dataframe = extractdata(file, loc, x, y)
    return dataframe


"""
The extractdata function takes a pandas dataframe, and the wound (x,y) location. It then converts the pandas labels to
the correct format for processing. It also normalises the cell trajectories to make sure that the wound is centered at
(0,0) making concatenation of multiple trajectories easier. It also adds a label to the cell trajectories to make sure
they are run in the correct order. The output of this is the normalised data frame. 
"""


def extractdata(file, loc, x, y):
    extracted_frame = pd.DataFrame(
        {'trackID': file['TRACK_ID'], 't': file['POSITION_T'], 'x': file['POSITION_X'], 'y': file['POSITION_Y']})
    extracted_frame['r'] = wound(extracted_frame['x'], extracted_frame['y'], x, y)
    extracted_frame['y'] = 353 - extracted_frame['y']
    """
    df['x'] = df['x'] - x # Can be uncommented to re-adjust the x,y co-ordinates\
     #to the make sure the wound is centered at (0,0)
    df['y'] = df['y'] - y
    """
    file['trackID'] = file['trackID'].astype(str)
    file.trackID = file.trackID + "{}".format(loc)  # creates a label for the tracks to be organised by
    return file


"""
The wound function, takes the cell x,y co-ordinates along with the wound locations and normalises the cell x,y with the
wound centered at 0,0. It then calculates the radial distance from the wound for each trajectory which is needed for the 
inference pipeline. Finally it we add this to the dataframe, and return the new radial distance. 
"""


def wound(dfx, dfy, wound_x, wound_y):
    xw = wound_x - dfx
    yw = wound_y - dfy
    r = np.sqrt(xw ** 2 + yw ** 2)
    return r


# Wound locations for mutant videos
x_wound = [224, 217, 221]
y_wound = [226, 121, 174]
# x_wound = [199,175,170,184,170,163,184]
# y_wound = [(353-234),(353-150),(353-107),(353-110),(353-238),(353-226),(353-220)]


# Wound locations for control videos
# x_wound = [140,193,170,163,148,181,172]
# y_wound = [140,(353-122),(353-170),(353-128),(353-128),(353-97),(353-120)]


# Creates the dataframe for the cell types (can be either mutant or control)
df1 = loadfile("1", "A", x_wound[1], y_wound[1])
df2 = loadfile("2", "B", x_wound[2], y_wound[2])
df3 = loadfile("3", "C", x_wound[3], y_wound[3])
df4 = loadfile("4", "D", x_wound[4], y_wound[4])
df5 = loadfile("5", "E", x_wound[5], y_wound[5])
df6 = loadfile("6", "F", x_wound[6], y_wound[6])
df7 = loadfile("7", "G", x_wound[7], y_wound[7])

# exclude dataframes 3 and 5 for mutant,they are the control for the mutant tissue
Concat_dataframes = [df1, df2, df3, df4, df5, df6, df7]

# The inference mechanism is dependent on the length of the array, if there are not enough tracks
# available in the array it cannot converge: (Uncomment if checking length of array)
# print("length of frame:", len(np.unique(df['trackID'])))

"""
space_slice splits the dataframe into different spatial bins. time_slice splits the data into different temporal bins,
this allows for the inference pipeline to calculate the spatial-temporal values for the bias, persistence, and weights. 
"""


def space_slice(df):
    s25 = df[(df['r'] >= 5) & (df['r'] <= 45)]
    s50 = df[(df['r'] >= 25) & (df['r'] <= 75)]
    s75 = df[(df['r'] >= 45) & (df['r'] <= 105)]
    s100 = df[(df['r'] >= 65) & (df['r'] <= 135)]
    s125 = df[(df['r'] >= 85) & (df['r'] <= 165)]
    s150 = df[(df['r'] >= 105) & (df['r'] <= 195)]
    s175 = df[(df['r'] >= 125) & (df['r'] <= 225)]
    return [s25, s50, s75, s100, s125, s150, s175]


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

distance = space_slice(Concat_dataframes)
space_time_dataframe = [time_slice(distance[i]) for i in range(len(distance))]

"""
This will run the inference method iteratively for each temporal and spatial bin and save
them as a numpy array for analysis in the data analysis Python script.
"""
# Error term to make sure we catch any issues where PointSource isn't used as the source


source = PointSource(position=np.array(0, 0))
if source != PointSource:
    print("Cannot run bias persistence inference pipeline, please check that source is PointSource")
else:
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
            inf_out = inferer.Ensembleinfer(n_walkers, n_iter)
            np.save(f'../data/WalkerData/PosterData/TwoWoundControlLoc3{i}{j}_new1', inf_out)
