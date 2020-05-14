#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:26:23 2020

@author: danieltudor
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))
import pickle
from inference.attractant_inference import AttractantInferer, observed_bias
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound
from utils.distributions import Normal

# Loads the np.array
out1 = np.load('../data/Control_data/Attractant Inference/MutantData.npy')

wound = PointWound(position=np.array([0, 0]))
## Plots the attractant inference data set

fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(12, 5), sharex='col')
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

names = ['$q$ [mol min$^{-1}$]', '$D$ [$\mu m^{2}$ min$^{-1}$]', 'τ [min]', '$R_0$ [mol $\mu m^{-2}$]', '$\kappa_d$ [mol $\mu m^{-2}$]', '$m$ [$\mu m^{2}$ mol$^{-1}$]', '$b_0$ [unitless]']
names1 = ['q', 'D', 'τ', 'R_0', 'kappa_d', 'm', 'b_0']

for j in range(7):
    data = out1[:,j]
    mean = np.mean(data,axis =0)
    std = np.std(data,axis =0)
    axes[j].set_title(names[j])
    axes[j].set_yticks([])
    axes[j].hist(out1[:, j], bins=50, color=cols[j], alpha=0.6, density=True,label='{} mean = {:.2f} $\pm$ {:.2f}'.format(names1[j],mean[j], std[j]))
    inferer.priors[j].plot(ax=axes[j], color=cols[j])

names = ['q', 'D', 'τ', 'R_0', 'kappa_d', 'm', 'b_0']
DataOutput= {}
for i in range(7):

    data = out1[:,i]
    mean = np.mean(data,axis =0)
    std = np.std(data,axis =0)
    print ("{} mean = ".format(names[i]), mean)
    print ("{} std = ".format(names[i]), std)

   # DataOutput[(mean[i], std[i])] = (mean,std)
