import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit



# Converts the csv file into a panda dataframe and updates to include the distance from the
# wound edge for spatial slicing
## C and E files are internal controls

def filename(name,loc,x,y):
    file = '../data/Mutant_stats_wound_{}_A_M.csv'.format(name)
    file  = pd.read_csv(file,header=0)
    df = dataframe(file,loc,x,y)
    return df

def dataframe(file,loc,x,y):
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


# Wound locations from videos
x_wound_m = [199,175,170,184,170,163,184]
y_wound_m = [(353-234),(353-150),(353-107),(353-110),(353-238),(353-226),(353-220)]

#Creates the dataframe for the mutant cell types
df1 = filename("one","A",x_wound_m[0],y_wound_m[0])
df2 = filename("two","B",x_wound_m[1],y_wound_m[1])
df3 = filename("three","C",x_wound_m[2],y_wound_m[2])
df4 = filename("four","D",x_wound_m[3],y_wound_m[3])
df5 = filename("five","E",x_wound_m[4],y_wound_m[4])
df6 = filename("six","F",x_wound_m[5],y_wound_m[5])
df7 = filename("seven","G",x_wound_m[6],y_wound_m[6])

#FramesMut = [df1,df2,df4,df6,df7] # exclude dataframes 3 and 5, they are the control for the mutant tissue
FrameContMut = [df3,df5]
#df = pd.concat(FramesMut)
df = pd.concat(FrameContMut) #Concatenates the control mutated data

# The inference mechanism is dependent on the length of the array, if there are not enough tracks
# available in the array it cannot converge
s25 = df[(df['r'] >= 5)  & (df['r'] <= 45)]
s50 = df[(df['r'] >= 25)  & (df['r'] <= 75)]
s75 = df[(df['r'] >= 45)  & (df['r'] <= 105)]
s100 = df[(df['r'] >= 65)  & (df['r'] <= 135)]
s125 = df[(df['r'] >= 85)  & (df['r'] <= 165)]
s150 = df[(df['r'] >= 105)  & (df['r'] <= 195)]
s175 = df[(df['r'] >= 125)  & (df['r'] <= 225)]
s200 = df[(df['r'] >= 145)  & (df['r'] <= 255)]

def time_slice(space):
    t10 = space[(space['t'] >=  0)  & (space['t'] <=600)]
    t17 = space[(space['t'] >=  180)  & (space['t'] <=1020)]
    t45 = space[(space['t'] >= 900)  & (space['t'] <= 2700)]
    t65 = space[(space['t'] >= 2100)  & (space['t'] <= 3900)]
    times = [t10,t17,t45,t65]
    return times


distance = [s25,s50,s75,s100,s125,s150,s175,s200]
s_distance = []
for i in range(len(distance)):
    s_distance.append(time_slice(distance[i]))


# This will run the inference method iteratively for each temporal and spatial bin and save
# them as a numpy array for analysis in the data analysis Python script

# This is important and needs to be changed to include an error note if PointWound is used instead of PointSource
from in_silico.sources import PointSource
source = PointSource(position=np.array([0, 0]))



from inference.walker_inference import BiasedPersistentInferer, prepare_paths
#for i in range(4):
for i in [0]:
    for j in [0]:
        inf = BiasedPersistentInferer(prepare_paths([paths[['x', 'y']].values for id, paths in s_distance[i][j].groupby('trackID')],include_t=False),source)
        inf_out = inf.multi_infer(n_walkers=6,n_steps=10000,burn_in=5000, suppress_warnings=True, use_tqdm  = True)
        np.save('../data/np_array/WB total control mutant-{}{}'.format(i,j),inf_out)
