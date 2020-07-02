"""
The below code runs as such:
1) Immune cell tracking data already processed by imageJ is inputted as a csv file and then processed using the pandas
   module to split the data set into time and space specific arrays
2) The inference method (BiasedPersistentInferer) is then called which runs the inference pipeline on each of the time/space arrays
3) Finally, it will save the individual numpy arrays for post-processing.

WT = Wild type data
Mut = Mutant data
"""
import sys
import os
sys.path.append(os.path.abspath('..'))
import pandas as pd
import numpy as np
from inference.walker_inference import BiasedPersistentInferer, prepare_paths
from in_silico.sources import PointSource
source = PointSource(position=np.array([0, 0]))

# Variables needed for Ensemble Monte Carlo
niter = 7500
nwalkers = 200

# Variables needed for Metroplis-Hastings Monte Carlo
nsteps = 500000
burn_in = 250000
n_walkers = 6


# Loads the file for either the mutant or wild-type csv files from imageJ
def filename(name,loc,x,y):
    fileWT = '../data/Control_stats_wound_{}_A_M.csv'.format(name)
    filemut = '../data/Mutant_stats_wound_{}_A_M.csv'.format(name)
    fileWT = pd.read_csv(fileWT,header=0)
    filemut = pd.read_csv(filemut,header=0)
    # Change from csv file to panda dataframes
    dfWT = TrajectoryData(fileWT,loc,x,y)
    dfmut = TrajectoryData(filemut,loc,x,y)
    return dfWT,dfmut

# Organise the dataframe
def TrajectoryData(file,loc,x,y):
    wound_x = x
    wound_y = y
    #Produce a new data frame with only the information needed for the inference pipeline
    df = pd.DataFrame({'trackID':file['TRACK_ID'],'t':file['POSITION_T'],'x':file['POSITION_X'],'y':file['POSITION_Y']})
    #Adjusts the y tracks, Fiji goes from top left to bottom left for y position, python goes
    #from bottom left to top left
    df['y'] = 353 - df['y']
    #Centres the tracks on a wound location of 0,0
    df['x'] = df['x'] - wound_x
    df['y'] = df['y'] - wound_y
    df['r'] = np.sqrt((wound_x - df['x'])**2 + (wound_y - df['y'])**2)
    df['trackID'] = df['trackID'].astype(str)
    #creates a label for the tracks to be organised by
    df.trackID = df.trackID + "{}".format(loc)
    return df

# WT wound locations from videos
x_wound_wt= [145,185,181,171,145,179,175]
y_wound_wt = [(353-219),(353-125),(353-118),(353-133),(353-127),(353-99),(353-113)]
# Mutant wound locations
x_wound_m = [199,175,170,184,170,163,184]
y_wound_m = [(353-234),(353-150),(353-107),(353-110),(353-238),(353-226),(353-220)]


#Creates the dataframe for the WT cell types
df1wt = filename("one","A",x_wound_wt[0],y_wound_wt[0])[0]
df2wt = filename("two","B",x_wound_wt[1],y_wound_wt[1])[0]
df3wt = filename("three","C",x_wound_wt[2],y_wound_wt[2])[0]
df4wt = filename("four","D",x_wound_wt[3],y_wound_wt[3])[0]
df5wt = filename("five","E",x_wound_wt[4],y_wound_wt[4])[0]
df6wt = filename("six","F",x_wound_wt[5],y_wound_wt[5])[0]
df7wt = filename("seven","G",x_wound_wt[6],y_wound_wt[6])[0]
#Creates the dataframe for the mutant cell types
df1m = filename("one","A",x_wound_m[0],y_wound_m[0])[1]
df2m = filename("two","B",x_wound_m[1],y_wound_m[1])[1]
df3m = filename("three","C",x_wound_m[2],y_wound_m[2])[1]
df4m = filename("four","D",x_wound_m[3],y_wound_m[3])[1]
df5m = filename("five","E",x_wound_m[4],y_wound_m[4])[1]
df6m = filename("six","F",x_wound_m[5],y_wound_m[5])[1]
df7m = filename("seven","G",x_wound_m[6],y_wound_m[6])[1]
#WT total
dfWT = pd.concat([df1wt,df2wt,df3wt,df4wt,df5wt,df6wt,df7wt])
#Mutant total (videos 3 and 5 are internal controls)
dfMut = pd.concat([df1m,df2m,df4m,df6m,df7m])
# Time and space binning of data sets
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

distance = space_slice(dfWT)
s_distance = [time_slice(distance[i]) for i in range(len(distance))]

# Runs the walker inference method from inference.walker_inference
# Initialises the count logger to 0
k = 0
time = s_distance[0]
for i in range(len(s_distance)):
    for j in range(len(time)):
        k += 1 # Tracks the number of bins
        print('analysing bin {}/{}'.format(k,(len(distance)*len(time))))# to give an overall sense of progress
        inferer = BiasedPersistentInferer(prepare_paths([paths[['x', 'y']].values for id, paths in s_distance[i][j].groupby('trackID')],include_t=False),source)
        inf_out = inferer.Ensembleinfer(nwalkers,niter)
        #MHOut = inferer.multi_infer(n_walkers,nsteps,burn_in,seed=0,suppress_warnings=True,use_tqdm=True)
        # Saves each time and space array as a numpy array for post processing
        np.save('../data/np_array/Walker WT-{}{}'.format(i,j),inf_out)
