import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
from inference.walker_inference import BiasedPersistentInferer, prepare_paths

# Variables needed for Ensemble Monte Carlo
niter = 1000
nwalkers = 70

# Variables needed for Metroplis-Hastings Monte Carlo
nsteps = 500000
burn_in = 250000
n_walkers = 6


# Converts the csv file into a panda dataframe and updates to include the distance from the
# wound edge for spatial slicing

## C and E files are internal controls

def LoadFile(name,loc,x,y):
    file = '/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/ImageJcsvs/Control_two_wound.csv'

    file  = pd.read_csv(file,header=0)
    df = ExtractData(file,loc,x,y)
    return df

def ExtractData(file,loc,x,y):
    x = x
    y = y
    df1 = file
    df = pd.DataFrame({'trackID':df1['TRACK_ID'],'t':df1['POSITION_T'],'x':df1['POSITION_X'],'y':df1['POSITION_Y']})
    df['r'] = wound(df['x'],df['y'],x,y)
    df['y'] = 353 - df['y']
    df['x'] = df['x'] - x
    df['y'] = df['y'] - y
    df['trackID'] = df['trackID'].astype(str)
    df.trackID = df.trackID + "{}".format(loc) #creates a label for the tracks to be organised by
    return df

def wound(dfx,dfy, x,y):
    wound_x = x
    wound_y = y
    xw = wound_x - dfx
    yw = wound_y - dfy
    r = np.sqrt((xw)**2 + (yw)**2)
    return r


# Wound locations for mutant videos
x_wound_m = [224,217,(224+217/2)]
y_wound_m = [226,121,(226+121/2)]
#x_wound_m = [199,175,170,184,170,163,184]
#y_wound_m = [(353-234),(353-150),(353-107),(353-110),(353-238),(353-226),(353-220)]
# Wound locations for control videos
#x_wound_m = [140,193,170,163,148,181,172]
#y_wound_m = [140,(353-122),(353-170),(353-128),(353-128),(353-97),(353-120)]
#Creates the dataframe for the mutant cell types
df = LoadFile("1","A",x_wound_m[0],y_wound_m[0])
#df2 = LoadFile("2","B",x_wound_m[1],y_wound_m[1])
#df3 = LoadFile("3","C",x_wound_m[3],y_wound_m[3])
#df4 = LoadFile("4","D",x_wound_m[5],y_wound_m[5])
#df5 = LoadFile("5","E",x_wound_m[6],y_wound_m[6])
#df6 = LoadFile("6","F",x_wound_m[5],y_wound_m[5])
#df7 = LoadFile("7","G",x_wound_m[6],y_wound_m[6])

#FramesCont = [df1,df2,df3,df4,df5,df6,df7] # exclude dataframes 3 and 5, they are the control for the mutant tissue
#df = pd.concat(FramesMut)
#df = pd.concat(FramesCont) #Concatenates the control data

#print("length of frame:", len(np.unique(df['trackID'])))
# The inference mechanism is dependent on the length of the array, if there are not enough tracks
# available in the array it cannot converge

def space_slice(df):
    s25 = df[(df['r'] >= 5)  & (df['r'] <= 45)]
    s50 = df[(df['r'] >= 25)  & (df['r'] <= 75)]
    s75 = df[(df['r'] >= 45)  & (df['r'] <= 105)]
    s100 = df[(df['r'] >= 65)  & (df['r'] <= 135)]
    s125 = df[(df['r'] >= 85)  & (df['r'] <= 165)]
    s150 = df[(df['r'] >= 105)  & (df['r'] <= 195)]
    s175 = df[(df['r'] >= 125)  & (df['r'] <= 225)]
    return [s25,s50,s75,s100,s125,s150,s175]

def time_slice(space):
    t5 = space[(space['t'] >=  0)  & (space['t'] <=600)]
    t15 = space[(space['t'] >=  180)  & (space['t'] <=1620)]
    t30 = space[(space['t'] >= 900)  & (space['t'] <= 2700)]
    t50 = space[(space['t'] >= 2100)  & (space['t'] <= 3900)]
    times = [t5,t15,t30,t50]
    return times



distance = space_slice(df)
s_distance = [time_slice(distance[i]) for i in range(len(distance))]
#test2 = s_distance[7][1]["trackID"].value_counts()

#print(test1), print(test2)
# This will run the inference method iteratively for each temporal and spatial bin and save
# them as a numpy array for analysis in the data analysis Python script

# This is important and needs to be changed to include an error note if PointWound is used instead of PointSource

from in_silico.sources import PointSource
source = PointSource(position=np.array([0, 0]))

k = 0
time = s_distance[0]
"""
for i in range(len(s_distance)):
    for j in range(len(time)):
        k += 1 # Tracks the number of bins
        print('analysing bin {}/{}'.format(k,(len(distance)*len(time))))# to give an overall sense of progress
        inferer = BiasedPersistentInferer(prepare_paths([paths[['x', 'y']].values for id, paths in s_distance[i][j].groupby('trackID')],include_t=False),source)
        inf_out = inferer.Ensembleinfer(nwalkers,niter)
       np.save('/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/TwoWoundControlloc1{}{}_new'.format(i,j),inf_out)
"""