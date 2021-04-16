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


# Converts the csv file into a panda dataframe and updates to include the distance from the
# wound edge for spatial slicing

## C and E files are internal controls

def LoadFile(name,loc,x,y):
    file = '/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/ImageJcsvs/Control_two_wound.csv'
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
    df['r'] = wound(df['x'],df['y'])
    df['angle'] = angles(df['x'],df['y'])
    df['angle'] = df['angle'].apply(lambda x: (2*np.pi) + x if x < 0 else x)
    df['trackID'] = df['trackID'].astype(str)
    df.trackID = df.trackID + "{}".format(loc) #creates a label for the tracks to be organised by
    return df

def wound(dfx,dfy):
    xw = dfx
    yw = dfy
    r = np.sqrt((xw)**2 + (yw)**2)
    return r

def angles(dfx,dfy):
    xw = dfx
    yw = dfy
    angle = np.arctan2(yw,xw)
    return angle

x1,x2= 208,218
xmp =  (x1+x2)/2
y1,y2 = 101,226
ymp = (y1+y2)/2
x_wound_m = [xmp]
y_wound_m = [ymp]

y_wound_m[:] = [353 - number for number in y_wound_m ]

#[119,114,123,124,92,115]
# [194,175,182,171,181]
# [240,152,107,229,217]
#Creates the dataframe for the mutant cell types
df = LoadFile("1","A",x_wound_m ,y_wound_m)

a1 = df[(df['angle'] >= (np.pi/4)) & (df['angle'] <((3*np.pi)/4))]
a2 = df[(df['angle'] >= ((5*np.pi)/4)) & (df['angle'] < ((7*np.pi)/4))]
a3 = df[(df['angle'] >= ((3*np.pi)/4)) & (df['angle'] < ((5*np.pi)/4))]
a4 = df[(df['angle'] >= ((7*np.pi)/4)) | (df['angle'] < ((np.pi)/4))]

s25 = df[(df['r'] >= 5)  & (df['r'] <= 45)]
s50 = df[(df['r'] >= 25)  & (df['r'] <= 75)]
s75 = df[(df['r'] >= 45)  & (df['r'] <= 105)]
s100 = df[(df['r'] >= 65)  & (df['r'] <= 135)]
s125 = df[(df['r'] >= 85)  & (df['r'] <= 165)]
s150 = df[(df['r'] >= 105)  & (df['r'] <= 195)]
s175 = df[(df['r'] >= 125)  & (df['r'] <= 225)]

fig, ax = plt.subplots(1,1,figsize=(5,5),sharex=True)

for ID, tracks in a1.groupby('trackID'):
        a1, = plt.plot(tracks['x'], tracks['y'],color = 'blue',lw=1)
for ID, tracks in a2.groupby('trackID'):
        a2, = plt.plot(tracks['x'], tracks['y'],color = 'red',lw=1)
for ID, tracks in a3.groupby('trackID'):
        a3, =plt.plot(tracks['x'], tracks['y'],color = 'green',lw=1)
for ID, tracks in a4.groupby('trackID'):
        a4, =plt.plot(tracks['x'], tracks['y'],color = 'black',lw=1)


#circle1=plt.Circle((208,(353 - 101)),15, color = 'grey')
#circle2=plt.Circle((218,(353 - 226)),15, color = 'grey')

#ax.add_artist(circle1)
#ax.add_artist(circle2)

plt.legend(handles = [a1,a2,a3,a4], 
           labels = ["First","Second","Third", "Fourth"],title = "Angle Bins", loc = [1,0.5])
plt.xlabel("X-distance ($\\mu m$)")
plt.ylabel("Y-distance ($\\mu m$)")
plt.title("Angle binning for immune cell trajectories")
plt.tight_layout()
plt.show()