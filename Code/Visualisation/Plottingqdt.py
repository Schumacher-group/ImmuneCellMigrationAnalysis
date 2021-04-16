import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt

WTOut =  np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/cont_delta.npy',allow_pickle=True)
WTsampler = WTOut[0]
outWT = WTsampler.get_chain(discard=100,thin =5, flat=True)


WTOut1 =  np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/cont_continuous.npy',allow_pickle=True)
WTsampler1 = WTOut1[0]
outWT1 = WTsampler1.get_chain(discard=100,thin =5, flat=True)



names = ['$q$ [mMol min$^{-1}$]', '$D$ [$\mu m^{2}$ min$^{-1}$]']#, '$R_0$ [mMol $\mu m^{-2}$]', '$\kappa_d$ [mMol $\mu m^{-2}$]', '$m$ [$\mu m^{2}$ mMol$^{-1}$]']

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5), sharex='col')
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig.suptitle("Mutant: Inferred chemoattractant parameters")

for j in range(len(names)):
    axes[j].set_title(names[j])
    axes[j].set_yticks([])
    axes[j].hist(outWT[:, j], bins=50, color=cols[j], alpha=0.6, density=True)#),label='{} mean = {:.2f} $\pm$ {:.2f}'.format(names1[j],mean[j], std[j]))
    axes[j].hist(outWT1[:, j], bins=50, color=cols[j], alpha=0.6, density=True)#),label='{} mean = {:.2f} $\pm$ {:.2f}'.format(names1[j],mean[j], std[j]))

