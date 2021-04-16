#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 12:13:17 2020

@author: danieltudor
"""
import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
import corner
from inference.attractant_inference import AttractantInferer
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound

WTOut = np.load('../data/DiffusionEqDeltaFc.npy',allow_pickle=True)
WTsampler = WTOut[0]
WTChain = WTsampler.get_chain(discard=5000,thin =5, flat=True)
MutOut = np.load('../data/AttractantInferenceMutantEmceeNew2Margin.npy',allow_pickle=True)
Mutsampler = MutOut[0]
MutChain= Mutsampler.get_chain(discard=5000,thin =20, flat=True)
"""
fig1 = plt.figure(facecolor='white')
ax1 = plt.axes(frameon=True)   
ax1.hist(samples[:,0],bins=50,alpha=0.6,density=True)
ax1.set(xlabel="Mmol / min")
ax1.axes.get_yaxis().set_visible(False)
plt.title("Posterior distribution: $q$")
"""
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(10, 4), sharex='col')
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

names = ['$M$', '$D$', '$R_0$', '$\kappa_d$', '$m$', '$b_0$']

for j in range(7):
        axes[j].set_title("{}".format(names[j]))
        axes[j].set_yticks([])
        #axes[j].hist(WTChain[:, j], bins=50, color=cols[j], alpha=0.6, density=True, label = "Mutant")
        axes[j].hist(WTChain[:, j], bins=50, color=cols[j], alpha=0.6, density=True, label = "WT")
        plt.legend()



plt.tight_layout()

"""
#fig = corner.corner(WTChain, labels=names,show_titles=True,quantiles=[0.05, 0.5, 0.95], title_kwargs={"fontsize": 12})

# import seaborn as sns
# q = out1[:,0]
# m = out1[:,5]
# ax1 = sns.jointplot(q,m,kind="hex",color='r')
# ax1.set_axis_labels(xlabel='q', ylabel='m', size ='large')

# fig, axes = plt.subplots(ncols=1, nrows=7, sharex=True)
# for i in range(7):
#     dist_out = WTChain
#     axes[i].plot(dist_out[:, i], linewidth=0.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], alpha=0.7)
#     axes[i].set_title(names[i])

# plt.tight_layout()

# plt.plot(WTChain[:,2])
"""