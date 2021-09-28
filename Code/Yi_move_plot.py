import os
import sys

sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
from inference.walker_inference_mixed_persistence import BiasedPersistentInferer
num_of_walkers = np.arange(15, 150, 15)

fig, ax = plt.subplots(3, 3, sharex = True)
ax = ax.flatten()
fig.suptitle("Inference of two persistences from mixed track data (equally mixed)")
for i, num in enumerate(num_of_walkers):
    EM_total = np.load(f"../data/Mixed_persistence_accuracy/Yi_data_total{i}_{num + num}.npy", allow_pickle=True)
    sampler = EM_total[0]
    p_total = sampler.get_chain(discard=1500, thin=1, flat=True)
    P1 = p_total[:, 2]
    P2 = p_total[:, 3]
    ax[i].hist(P1, bins=100, alpha=0.6, density=True)
    ax[i].hist(P2, bins=100, alpha=0.6, density=True)
    ax[i].axvline(0.3, color='black', ls='--', label="True value: 0.3")
    ax[i].axvline(0.7, color='black', ls='--', label="True value: 0.7")
    ax[i].set_title(f'Track num:{num+num}')
#/Users/danieltudor/Documents/GitHub/ImmuneCellMigrationAnalysis/data/Mixed_persistence_accuracy/Yi_data_total0_30.npy

fig.text(0.5, 0.01, 'Persistence', ha='center')

plt.legend()
plt.tight_layout()
plt.savefig("../data/Mixed_persistence_accuracy/Persistence_total_mix_w2_p1_p2_all_tracks.pdf")
plt.show()