import sys
import os

sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
from inference.walker_inference import BiasedPersistentInferer, prepare_paths
from in_silico.sources import PointSource
# Variables needed for Ensemble Monte Carlo
niter = 2000
nwalkers = 10

# Variables needed for Metroplis-Hastings Monte Carlo
nsteps = 500000
burn_in = 250000
n_walkers = 6


# Converts the csv file into a panda dataframe and updates to include the distance from the
# wound edge for spatial slicing

## C and E files are internal controls
"""
Functions for extracting the tracks from the raw csv files outputted by ImageJ. 
Function definitions:
---------------------
LoadFile = loads the ImageJ csv file with the raw data sets 
dataframe = extracts the data needed for the inference, i.e. x,y,trackID,t 
wound = adds an extra column to the dataframe which allows for the radius from the wound to be included
angles = converts the location of the immune cells from Cartesian coordinates to polar coordinates  

"""
def LoadFile(name,loc,x,y):
    file = f'/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/ImageJcsvs/Control_{name}_new.csv'
    file = pd.read_csv(file,header=0)
    df = create_dataframe(file,x,y,loc)
    return df

def create_dataframe(df, xw, yw,loc):
    reshapeddata = pd.DataFrame({'trackID':df['TRACK_ID'],'t':df['POSITION_T'],'x':df['POSITION_X'],'y':df['POSITION_Y']})
    reshapeddata['x'] = reshapeddata['x'] - xw
    reshapeddata['y'] = reshapeddata['y'] - yw
    reshapeddata['r'] = (lambda x, y: np.sqrt(x ** 2 + y ** 2))(reshapeddata['x'], reshapeddata['y'])
    reshapeddata['trackID'] = reshapeddata['trackID'].astype(str)
    reshapeddata.trackID = reshapeddata.trackID + "{}".format(loc)  # creates a label for the tracks to be organised by
    return reshapeddata




x1,x2= 208,218
xmp =  (x1+x2)/2
y1,y2 = 101,226
ymp = (y1+y2)/2
x_wound_m = [145,185,181,171,145,179,175]
y_wound_m = [(353-219),(353-125),(353-118),(353-133),(353-127),(353-99),(353-113)]


#y_wound_m[:] = [353 - number for number in y_wound_m ]

#[119,114,123,124,92,115]
# [194,175,182,171,181]
# [240,152,107,229,217]
#Creates the dataframe for the mutant cell types
#df = LoadFile("1","A",x_wound_m ,y_wound_m)


df1 = LoadFile("1","A",x_wound_m[0],y_wound_m[0])
df2 = LoadFile("2","B",x_wound_m[1],y_wound_m[1])
df3 = LoadFile("3","C",x_wound_m[2],y_wound_m[2])
df4 = LoadFile("4","D",x_wound_m[3],y_wound_m[3])
df5 = LoadFile("5","E",x_wound_m[4],y_wound_m[4])
df6 = LoadFile("6","F",x_wound_m[5],y_wound_m[5])
df7 = LoadFile("7","G",x_wound_m[6],y_wound_m[6])
""""
dfu1 = LoadFile("1","A",0,0)
dfu2 = LoadFile("2","A",0,0)
"""
FramesCont = [df1,df2,df3,df4,df5,df6,df7] # exclude dataframes 3 and 5, they are the control for the mutant tissue
#FramesCont = [dfu1,dfu2]
trajectory = pd.concat(FramesCont) #Concatenates the control data
# The inference mechanism is dependent on the length of the array, if there are not enough tracks
# available in the array it cannot converge



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
    t20 = space[(space['t'] >= 0) & (space['t'] <= (20 * 60))]
    t35 = space[(space['t'] >= (20 * 60)) & (space['t'] <= (35 * 60))]
    t50 = space[(space['t'] >= (35 * 60)) & (space['t'] <= (50 * 60))]
    t65 = space[(space['t'] >= (50 * 60)) & (space['t'] <= (65 * 60))]

    return [t20, t35, t50, t65]


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
            prepare_paths([paths[['x', 'y']].values for id, paths in s_distance[i][j].groupby('trackID')],
                          include_t=False), source)
        inf_out = inferer.ensembleinfer(NWalkers, NIters)
        np.save(
            '/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/WalkerData/WildType_new{}{}'.format(
                i, j), inf_out)