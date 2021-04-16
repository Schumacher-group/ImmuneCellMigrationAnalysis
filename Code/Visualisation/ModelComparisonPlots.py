#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 17:09:27 2020

@author: danieltudor
"""

import os
import sys
sys.path.append(os.path.abspath('..'))
from inference.attractant_inference import AttractantInferer,observed_bias
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound
import numpy as np
import matplotlib.pyplot as plt
## This takes in the Random Walker inferred parameters and outputs a dictionary with the observed bias parameters
## Variables
# Variables needed for Ensemble Monte Carlo
niter = 10000
nwalkers = 100

#Chose wound locations
wound_centre = PointWound(position=np.array([0, 0]))
#wound_inside = CellsInsideWound(position=np.array([0, 0]),n_cells=10,radius=15)
wound_margin = CellsOnWoundMargin(centre=np.array([0,0]),n_cells=10,radius=5)

#Load data from numpy arrays
DiffNoProd = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/WildTypeData.npy',allow_pickle=True)
DiffProdTimesampler = DiffNoProd[0]
samples = DiffProdTimesampler.get_chain(discard=5000,thin =20, flat=True)

def Bias_persistance(x,y):
    WT = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/UnwoundedControlData-{}{}.npy'.format(x,y),allow_pickle=True)
    samplerWT = WT[0]
    WildTypeChain = samplerWT.get_chain(discard=250,thin = 2,flat=True)
    W =  WildTypeChain[:,0]
    B =  WildTypeChain[:,2]
    OBWT = (W * B)
    return OBWT


def OBSIZE(val):
        OBWT = []
        for i in range(7):
            DataSet =  Bias_persistance(i,val)
            WT = DataSet
            OBWT.append(WT)
        return OBWT

fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True)
    
# instantiate a point wound
wound1 = CellsOnWoundMargin(centre=np.array([0,0]),n_cells=20, radius=20)

r_points = np.array([25, 50, 75, 100, 125, 150,175])


# where to measure observed bias
r = np.linspace(25, 180, 100)
t = np.array([5,15,30,50])

for ax, p in zip(axes, t):
    ax.set_ylabel('$t={}$'.format(p), rotation=0, size='large', labelpad=35)
lines = []
scatters = []
for i, tt in enumerate(t):
    OB = OBSIZE(i)
    MeanWT = np.mean(OB,axis=1)
    stdWT = np.std(OB,axis=1)
    for theta in samples[np.random.randint(len(samples), size=100)]:
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        ob = observed_bias(theta, r, tt, wound1)
        lines.append(axes[i].plot(r, ob, color=col, linewidth=1,alpha=0.1)[0])
        axes[i].set_ylim(0, 0.3)
    axes[i].errorbar(r_points,MeanWT,yerr = stdWT,fmt='r.',capsize=5, label = "Data:Observed bias")

axes[0].set_title('Observed bias')
axes[-1].set_xlabel('Distance ($\\mu m$)')
plt.legend(loc=[0.6,4.2])
plt.tight_layout()
