#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:14:45 2020

@author: danieltudor
"""


import sys
import os
sys.path.append(os.path.abspath('..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def LoadFile(name,loc,x,y):
    file = '/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/ImageJcsvs/Control_{}_new.csv'.format(name)
    file  = pd.read_csv(file,header=0)
    df = dataframe(file,loc,x,y)
    return df

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


x_wound_m = [148,191,172,163,151,184,178]
y_wound_m = [208,119,114,123,124,92,115]
y_wound_m[:] = [353 - number for number in y_wound_m ]

#Creates the dataframe for the mutant cell types
df1 = LoadFile("1","A",x_wound_m[0],y_wound_m[0])
df2 = LoadFile("2","B",x_wound_m[1],y_wound_m[1])
df3 = LoadFile("3","C",x_wound_m[3],y_wound_m[3])
df4 = LoadFile("4","D",x_wound_m[5],y_wound_m[5])
df5 = LoadFile("5","E",x_wound_m[6],y_wound_m[6])
df6 = LoadFile("6","F",x_wound_m[5],y_wound_m[5])
df7 = LoadFile("7","G",x_wound_m[6],y_wound_m[6])

FramesCont = [df1,df2,df3,df4,df5,df6,df7] # exclude dataframes 3 and 5, they are the control for the mutant tissue
#df = pd.concat(FramesMut)
df = pd.concat([df1,df2,df3,df4,df5,df6,df7]) 

s25 = df[(df['r'] >= 5)  & (df['r'] <= 45)]
s50 = df[(df['r'] >= 25)  & (df['r'] <= 75)]
s75 = df[(df['r'] >= 45)  & (df['r'] <= 105)]
s100 = df[(df['r'] >= 65)  & (df['r'] <= 135)]
s125 = df[(df['r'] >= 85)  & (df['r'] <= 165)]
s150 = df[(df['r'] >= 105)  & (df['r'] <= 195)]
s175 = df[(df['r'] >= 125)  & (df['r'] <= 225)]

fig, ax = plt.subplots(1,1,figsize=(5,5),sharex=True)
for ID, tracks in s175.groupby('trackID'):
        s175, = ax.plot(tracks['x'], tracks['y'],color = 'xkcd:light pink',lw=1)
for ID, tracks in s150.groupby('trackID'):
        s150, = plt.plot(tracks['x'], tracks['y'],color = 'xkcd:charcoal',lw=1)
for ID, tracks in s125.groupby('trackID'):
        s125, =plt.plot(tracks['x'], tracks['y'],color = 'xkcd:rust',lw=1)
for ID, tracks in s100.groupby('trackID'):
        s100, =plt.plot(tracks['x'], tracks['y'],color = 'xkcd:pine green',lw=1)
for ID, tracks in s75.groupby('trackID'):
        s75, =plt.plot(tracks['x'], tracks['y'],color = 'xkcd:salmon',lw=1)
for ID, tracks in s50.groupby('trackID'):
        s50, =plt.plot(tracks['x'], tracks['y'],color = 'xkcd:sky blue',lw=1)
for ID, tracks in s25.groupby('trackID'):
        s25, = plt.plot(tracks['x'], tracks['y'],color = 'xkcd:sage',lw=1)
plt.legend(handles = [s175,s150,s125,s100,s75,s50,s25], 
           labels = ["125$\\mu m$ - 225$\\mu m$","105$\\mu m$ - 195$\\mu m$",
           "85$\\mu m$ - 165$\\mu m$","65$\\mu m$ - 135$\\mu m$",
           "45$\\mu m$ - 105$\\mu m$","25$\\mu m$ - 75$\\mu m$",
           "5$\\mu m$ - 45$\\mu m$"],title = "Spatial Bins", loc = [1,0.5])
plt.xlabel("X-distance ($\\mu m$)")
plt.ylabel("Y-distance ($\\mu m$)")
plt.title("Spatial binning for immune cell trajectories")
plt.tight_layout()