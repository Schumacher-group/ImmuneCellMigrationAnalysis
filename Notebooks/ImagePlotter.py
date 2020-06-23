#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:32:31 2020

@author: danieltudor
"""


import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inference.attractant_inference import AttractantInferer, observed_bias
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound
from utils.distributions import Normal
import seaborn as sns

outNW = np.load('../data/np_array/Unwounded-data/AttractantInferenceNWNoB.npy')
outMut = np.load('../data/np_array/Unwounded-data/AttractantInferenceMutNoB.npy')
outWT = np.load('../data/np_array/Unwounded-data/AttractantInferenceWTNoB.npy')
def Bias_persistance(x,y):
    Dataset = np.load('../data/np_array/WB total WT-{}{}.npy'.format(x,y))
    W = Dataset[:,0]
    B = Dataset[:,2]
    OB = (W * B)
    mean = np.mean(OB,axis=0)
    std = np.std(OB,axis=0)

    return mean,std

plt.axhspan(0,0.3,0.83,1,facecolor = "black",alpha=0.2)
plt.text(52, 0.25, "no data")
sns.distplot(outWT[:,2], label = "Wild type")
sns.distplot(outNW[:,2], label = "Unwounded")
sns.distplot(outMut[:,2], label = "Mutant")
plt.plot([50, 50], [0, 0.3], 'k--', lw=2, label = "location of final time bin")
plt.xlim(0,60)
plt.ylim(0,0.3)


plt.xlabel("Production time ($\\tau$)")
plt.ylabel("Density")
plt.legend()
plt.savefig("../Images/All production time No B .pdf")


sns.distplot(outWT[:,1], label = "Wild type")
sns.distplot(outNW[:,1], label = "Unwounded")
sns.distplot(outMut[:,1], label = "Mutant")
plt.xlabel("Diffusion coefficient ($D$)")
plt.ylabel("Density")
plt.legend()
plt.savefig("../Images/All Diffusion coefficient No  B.pdf")

sns.distplot(outWT[:,3], label = "Wild type")
sns.distplot(outNW[:,3], label = "Unwounded")
sns.distplot(outMut[:,3], label = "Mutant")
plt.xlabel("Receptor concentration ($R_0$)")
plt.ylabel("Density")
plt.legend()
plt.savefig("../Images/R0 No  B.pdf")


sns.distplot(outWT[:,4], label = "Wild type")
sns.distplot(outNW[:,4], label = "Unwounded")
sns.distplot(outMut[:,4], label = "Mutant")
plt.xlabel("Dissociation constant ($\kappa_d$)")
plt.ylabel("Density")
plt.legend()
plt.savefig("../Images/Kappa d.pdf")

sns.distplot(outWT[:,0], label = "Wild type")
sns.distplot(outNW[:,0], label = "Unwounded")
sns.distplot(outMut[:,0], label = "Mutant")
plt.xlabel("Secretion rate ($q$)")
plt.ylabel("Density")
plt.legend()
plt.savefig("../Images/q.pdf")