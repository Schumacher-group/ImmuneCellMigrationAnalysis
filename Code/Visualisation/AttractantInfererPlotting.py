
# # Attractant Dynamics Parameters

import os
import sys
sys.path.append(os.path.abspath('..'))
from inference.attractant_inference import AttractantInferer
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound
import numpy as np
import matplotlib.pyplot as plt

EnsembleOut = np.load('../data/AttractantInferenceEnsembleWT.npy',allow_pickle=True)
MHOut = np.load('../data/AttractantInferenceMHWT.npy',allow_pickle=True)
# Needed to output emcee parameters
sampler = EnsembleOut[0]
samples = sampler.flatchain()

# Compare parameter outputs between emcee and MH
import seaborn as sns
sns.distplot(samples[:,1], label = "emcee")
sns.distplot(MHOut[:,1], label = "MH")
plt.xlabel("Production time ($\\tau$)")
plt.ylabel("Density")
plt.legend()


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
