# allow imports from the LM package
import os
import sys
sys.path.append(os.path.abspath('../..'))
from inference.attractant_inference import AttractantInferer, observed_bias
from in_silico.sources import PointWound
import numpy as np
import matplotlib.pyplot as plt
# Parameters for model 1 and to capture the bias of the cells
q= 400
D= 100
tau = 35
R0=0.3
kappa=0.3
m=  8
b0 = 0.01
params = np.array([q,D,tau,R0,kappa, m, b0])


# Parameters for tuning model 2 to model 1
params_model2 = np.array([10,50,R0,kappa, m, b0])


fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True)

# instantiate a point wound
wound = PointWound(position=np.array([0, 0]))

# where to measure observed bias
r_points = np.array([25, 50, 75, 100, 125, 150, 175])
r = np.linspace(25, 175, 100)
t = np.array([5, 15, 30,50])


for ax, p in zip(axes, t):
    ax.set_ylabel('$t={}$'.format(p), rotation=0, size='large', labelpad=35)

# plot the points
lines = []
scatters = []
for i, tt in enumerate(t):
    col = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
    lines.append(axes[i].plot(r, observed_bias(params_model2, r, tt, wound), color=col, linewidth=1)[0])
    scatters.append(axes[i].plot(r_points, observed_bias(params, r_points, tt, wound), color=col, marker='o', linewidth=0, markersize=4)[0])
    #lines.append(axes[i].plot(r, observed_bias(params,r, tt, wound), color=col, marker='o', linewidth=0, markersize=4)[0])
    axes[i].set_ylim(0, 1)

axes[0].set_title('A point wound: observed bias as a function of distance')
axes[-1].set_xlabel('Distance, microns')
plt.tight_layout()


# get the observed bias readings from the sliders graph.
# ob_readings is a dictionary containing {(r, t): mu, sig} where mu and sig are the observed bias mean and standard deviation at the point r and time t
np.random.seed(100)
ob_readings = {}
for T, ob in zip(t, scatters):
    mus = ob.get_ydata()
    rs = ob.get_xdata()
    for r, mu in zip(rs, mus):
        ob_readings[(r, T)] = (mu, 0.02)

#ob_readings

niter = 1000
#nwalkers = 100

nsteps = 50000
burn_in = 25000
n_walkers = 6
"""
# This runs the inference using model 1
Model1Inferer = AttractantInferer(ob_readings, wound=wound, t_units='minutes',dynamics = 0)
Model1 = Model1Inferer.multi_infer(n_walkers,nsteps,burn_in,seed=0,suppress_warnings=True,use_tqdm=True)
np.save('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/HM1Data',Model1)

"""
Model2Inferer = AttractantInferer(ob_readings, wound=wound, t_units='minutes',dynamics = 1)
Model2 = Model2Inferer.multi_infer(n_walkers,nsteps,burn_in,seed=0,suppress_warnings=True,use_tqdm=True)
np.save('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/HM2Data',Model2)
