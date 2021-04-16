#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:37:11 2020

@author: danieltudor
"""

import os
import sys
sys.path.append(os.path.abspath('..'))
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from utils.distributions import Normal
from in_silico.walkers import BP_Leukocyte
from in_silico.sources import PointSource
from utils.plotting import plot_paths

from inference.walker_inference import BiasedPersistentInferer


total_p= np.load("/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/persistences.npy", allow_pickle = True)
sampler = total_p[0]


p_total = sampler.get_chain(discard=100, thin=10, flat=True)
P = p_total[:,1]
plt.hist(P,bins=100,alpha=0.6,density=True)

plt.axvline(0.3,color='black',ls='--', label = "True value: 0.3")
plt.axvline(0.7,color='black',ls='--', label = "True value: 0.7")
plt.xlabel("Persistence parameter",fontsize=16)
plt.ylabel("Probability Density",fontsize=16)
plt.xlim(0.2,0.8)
plt.legend()

"""
p_total = sampler.get_chain(discard=0, thin=10, flat=True)
P1 = p_total[:,1]
plt.plot(P1)
plt.text(10000, 0.95, 'Number of ensemble walkers = 800, Iteratons = 1000', style='italic',
        bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 10})
plt.axhspan(0,1,0.1,0.04,facecolor = "black",alpha=0.2)
plt.annotate('burn-in', xy=(5000, 0.6), xytext=(35000, 0.7),
            arrowprops=dict(facecolor='black', shrink=0.01))
plt.xlabel("Iterations")
plt.title("Persistence emcee chain")

"""