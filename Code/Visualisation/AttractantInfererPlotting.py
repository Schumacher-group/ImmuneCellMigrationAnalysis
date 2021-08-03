
# # Attractant Dynamics Parameters

import os
import sys
sys.path.append(os.path.abspath('..'))


import numpy as np
import matplotlib.pyplot as plt
from inference.attractant_inference import AttractantInferer
from in_silico.sources import PointWound
import numpy as np
EnsembleOut = np.load('/Users/danieltudor/Documents/GitHub/ImmuneCellMigrationAnalysis/data/Emcee_posterior_chain_control_SMBData.npy',allow_pickle=True)
#MHOut = np.load('../data/AttractantInferenceMHWT.npy',allow_pickle=True)
# Needed to output emcee parameters
sampler = EnsembleOut[0]
samples = sampler.get_chain(discard=250,thin = 2,flat=True)

# Compare parameter outputs between emcee and MH
import seaborn as sns


f, axs = plt.subplots(1, 3, figsize=(8, 4))

sns.histplot(samples[:,0],x ="Flow rate", kde= True, ax=axs[0])
sns.histplot(samples[:,1],x ="Diff Co", kde= True, ax=axs[1])
sns.histplot(samples[:,2],x ="Prod time", kde= True, ax=axs[2])
plt.legend()
f.tight_layout()

#sns.distplot(MHOut[:,1], label = "MH")
plt.savefig("/Users/danieltudor/Desktop/Image_1.png")

"""
# Outputs the chains for each parameters
fig, axes = plt.subplots(7, figsize=(10, 7), sharex=True)
labels = ["q", "D", "$\\tau$", "$\kappa_d$","R_0","m","b_0"]
for i in range(len(labels)):
    ax = axes[i]
    ax.plot(samples[:,i], alpha=1)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")
"""
