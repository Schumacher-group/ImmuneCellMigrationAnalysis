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
WTOut = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/WildTypeAttractantData.npy',allow_pickle=True)
WTsampler = WTOut[0]
outMut = WTsampler.get_chain(discard=5000,thin =15, flat=True)


mutant1 = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/DiffusionNoWound1.npy',allow_pickle=True)
Mutant1= mutant1[0]
outMut1 = Mutant1.get_chain(discard=500,thin =5, flat=True)
names = ['$D$ [$\mu m^{2}$ min$^{-1}$]', 'τ [min]']#, '$R_0$ [mMol $\mu m^{-2}$]', '$\kappa_d$ [mMol $\mu m^{-2}$]', '$m$ [$\mu m^{2}$ mMol$^{-1}$]','b0']
names1 = ['q','D', 'τ']
priors  = [10000,60]#,1,1,30,0.5]
col = ['r','b','g']
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 25))
fig.suptitle("Mutant and unwounded:Inferred chemoattractant parameters",fontsize = 18)
sns.set_palette("muted")
for j in range(len(names)):
    axes[j].set_xlim(0,priors[j])
    axes[j].set_title(names[j],fontsize = 15)
    axes[j].set_yticks([])
    #sns.distplot(outMut1[:, j+1], hist=False, kde_kws={"shade": True}, label = 'Unwounded', ax=axes[j])
    sns.distplot(outMut[:, j+1], hist=False, kde_kws={"shade":True}, label = 'Mutant', ax=axes[j])

    #sns.distplot(outWT[:, j], hist=False,kde_kws={"shade": False}, label = 'No wound', ax=axes[j])
   
bbox = dict(boxstyle="round", fc="0.8")
arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle,angleA=0,angleB=90,rad=10")

offset =  80
axes[1].annotate('Final time bin= {} mins'.format(50),
            (50,0), xytext=(-2*offset, offset), textcoords='offset points',
            bbox=bbox, arrowprops=arrowprops,fontsize = 12)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])