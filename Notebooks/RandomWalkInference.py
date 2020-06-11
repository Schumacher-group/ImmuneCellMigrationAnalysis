
import sys
import os
sys.path.append(os.path.abspath('..'))
import pandas as pd
import numpy as np
from inference.walker_inference import BiasedPersistentInferer, prepare_paths
from in_silico.sources import PointSource

def filename(name,loc,x,y):
    filec = '../data/Control_stats_wound_{}_A_M.csv'.format(name)
    #filemut = '../data/Mutant_stats_wound_{}_A_M.csv'.format(name)
    filec  = pd.read_csv(filec,header=0)
    #filemut = pd.read_csv(filemut,header=0)

    dfcont = dataframe(filec,loc,x,y)
    #dfmut = dataframe(filemut,loc,x,y)

    return dfcont#,dfmut

def dataframe(file,loc,x,y):
    x = x
    y = y
    df1 = file
    df = pd.DataFrame({'trackID':df1['TRACK_ID'],'t':df1['POSITION_T'],'x':df1['POSITION_X'],'y':df1['POSITION_Y']})
    df['y'] = 353 - df['y']
    df['x'] = df['x'] - x
    df['y'] = df['y'] - y
    df['r'] = wound(df['x'],df['y'],x,y)
    df['trackID'] = df['trackID'].astype(str)
    df.trackID = df.trackID + "{}".format(loc) #creates a label for the tracks to be organised by
    return df

def wound(dfx,dfy, x,y):
    wound_x = 0
    wound_y = 0
    xw = wound_x - dfx
    yw = wound_y - dfy
    r = np.sqrt((xw)**2 + (yw)**2)
    return r


# WT wound locations from videos
x_wound_c= [145,185,181,171,145,179,175]
y_wound_c = [(353-219),(353-125),(353-118),(353-133),(353-127),(353-99),(353-113)]
# Mutant wound locations
x_wound_m = [199,175,170,184,170,163,184]
y_wound_m = [(353-234),(353-150),(353-107),(353-110),(353-238),(353-226),(353-220)]


#Creates the dataframe for the WT cell types
df1c = filename("one","A",x_wound_c[0],y_wound_c[0])
df2c = filename("two","B",x_wound_c[1],y_wound_c[1])
df3c = filename("three","C",x_wound_c[2],y_wound_c[2])
df4c = filename("four","D",x_wound_c[3],y_wound_c[3])
df5c = filename("five","E",x_wound_c[4],y_wound_c[4])
df6c = filename("six","F",x_wound_c[5],y_wound_c[5])
df7c = filename("seven","G",x_wound_c[6],y_wound_c[6])
#Creates the dataframe for the mutant cell types
df1m = filename("one","A",x_wound_m[0],y_wound_m[0])
df2m = filename("two","B",x_wound_m[1],y_wound_m[1])
df3m = filename("three","C",x_wound_m[2],y_wound_m[2])
df4m = filename("four","D",x_wound_m[3],y_wound_m[3])
df5m = filename("five","E",x_wound_m[4],y_wound_m[4])
df6m = filename("six","F",x_wound_m[5],y_wound_m[5])
df7m = filename("seven","G",x_wound_m[6],y_wound_m[6])
#WT totle
dfWT = pd.concat([df1c,df2c,df3c,df4c,df5c,df6c,df7c])
#Male WT
dfWTM = pd.concat([df1c,df3c,df4c,df7c])
#Female WT
dfWTF = pd.concat([df2c,df5c,df6c])
#Mutant total
dfMut = pd.concat([df1m,df2m,df4m,df6m,df7m])
#Mutant Male
dfMutM = pd.concat([df1m,df2m,df7m])
#Mutant Female
dfMutF = pd.concat([df4m,df6m])
#Mutant Control
dfMutCont = pd.concat([df3m,df5m])
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
#Types = [dfWTM,dfWTF,dfMutM,dfMutF,dfMutCont]
#Gender = ["WT-Male","WT-Female","Mut-Male","Mut-Female","Mut-Cont"]
distance = space_slice(dfWT)
s_distance = []
for i in range(len(distance)):
    s_distance.append(time_slice(distance[i]))
k = 0
time = s_distance[0]
for i in range(len(s_distance)):
    for j in range(len(times)):
        k += 1 # Tracks the number of bins
        print('analysing bin {}/{}{}'.format(k,n,(len(distance)*len(s_distance[0]))))# to give an overall sense of progress
        inf = BiasedPersistentInferer(prepare_paths([paths[['x', 'y']].values for id, paths in s_distance[i][j].groupby('trackID')],include_t=False), PointSource((0,0)))
        inf_out = inf.multi_infer(n_walkers=10,n_steps=15000,burn_in=5000,step=0, suppress_warnings=True, use_tqdm  = True)
        np.save('../data/np_array/Walker WT-{}{}'.format(,i,j),inf_out)
