"""
Modelling of two persistence model using synthetic data, to check for the required number of movies to accurately
capture the two persistence's within the dataset
"""

import os
import sys
sys.path.append(os.path.abspath('..'))
import matplotlib.pyplot as plt
import numpy as np
from in_silico.walkers import BP_Leukocyte
from in_silico.sources import PointSource

# make a source (for the walkers to migrate towards)
source = PointSource(position=np.array([0, 0]))
"""
Paths for tracks created using persistence = 0.7
"""

# instantiate a walker with parameters w=0.5, p=0.7, b=0.7
w, p, b = 0.5, 0.7, 0.7
params = np.array([w, p, b])
walker = BP_Leukocyte(params, source)

# these are the starting positions of the leukocytes. Size must be of the form (N, 2)
X0s = np.random.uniform(-5, 5, size=(100, 2))
paths_p07 = walker.walk(X0s, T=50)
"""
Paths for tracks created using persistence = 0.3
"""

# instantiate a walker with parameters w=0.5, p=0.3, b=0.7
w, p, b = 0.5, 0.3, 0.7
params = np.array([w, p, b])
walker = BP_Leukocyte(params, source)

# these are the starting positions of the leukocytes. Size must be of the form (N, 2)
X0s = np.random.uniform(-5, 5, size=(100, 2))
paths_p03 = walker.walk(X0s, T=50)


print('The paths object has shape {}: (T+1, [x, y], n_walkers)'.format(paths_p03.shape))


# Concatenate the two path arrays together to make one dataset with mixed persistence values

total_paths = np.concatenate((paths_p03, paths_p07), axis=2)


# Run inference on the mixed persistence's,  save and plot the results
from inference.walker_inference_mixed_persistence import BiasedPersistentInferer
inferer_total = BiasedPersistentInferer(total_paths, source)
niter = 2000 # number of MCMC iterations
nwalkers = 140 # number of
EM_total = inferer_total.ensembleinfer(nwalkers,niter)


sampler = EM_total[0]
p_total = sampler.get_chain(discard=1000, thin=1, flat=True)
P1 = p_total[:,2]
P2 = p_total[:,3]
np.save("/Users/danieltudor/Documents/GitHub/ImmuneCellMigrationAnalysis/data/Mixed_persistence_accuracy/EM_total",EM_total)
plt.hist(P1,label='$P$ = {:.2f} $\pm$ {:.2f}'.format(np.mean(P1), np.std(P1)),bins=100,alpha=0.6,density=True)
plt.hist(P2,label='$P$ = {:.2f} $\pm$ {:.2f}'.format(np.mean(P2), np.std(P2)),bins=100,alpha=0.6,density=True)

plt.axvline(0.3,color='black',ls='--', label = "True value: 0.3")
plt.axvline(0.7,color='black',ls='--', label = "True value: 0.7")
plt.xlabel("Persistence parameter",fontsize=16)
plt.ylabel("Probability Density",fontsize=16)
plt.legend()
plt.savefig("/Users/danieltudor/Documents/GitHub/ImmuneCellMigrationAnalysis/data/Mixed_persistence_accuracy/Persistence_total_mix_w2_p1_p2.pdf")
plt.show()




