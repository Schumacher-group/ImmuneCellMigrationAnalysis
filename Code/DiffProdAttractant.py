import os
import sys
sys.path.append(os.path.abspath('..'))
from in_silico.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inference.attractant_inference import AttractantInferer, observed_bias
from in_silico.sources import CellsOnWoundMargin, PointWound, CellsInsideWound
# Attractant inference
# Variables needed for Ensemble Monte Carlo
niter = 1000
nwalkers = 100


wound = CellsOnWoundMargin(centre=np.array([0, 0]))
"""
def Bias_persistance(x,y):
    Dataset = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/WildTypeData-{}{}.npy'.format(x,y),allow_pickle=True)
    sampler = Dataset[0]
    Dataset = sampler.get_chain(discard=250,thin = 2,flat=True)
    W = Dataset[:,0]
    B = Dataset[:,2]
    OBWT = (W * B)
    return OBWT

distance = [25,50,75,100,125,150,175]
time = [5,10,30,50]
means=[]
ob_readings = {}
for i in range(len(time)):
    for j in range(len(distance)):
        data = Bias_persistance(j,i)
        mean = np.mean(data)
        std = np.std(data)
        means.append(mean)
        ob_readings[(distance[j], time[i])] = (mean,std)



        # Attractant inference
inferer = AttractantInferer(ob_readings, wound=wound, t_units='minutes')

        # Emcee (Ensemble Monte Carlo)
EnsembleOut = inferer.Ensembleinfer(nwalkers,niter)
#DiffProdTime = np.save('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/DiffusionMutant1',EnsembleOut)
#EnsembleOut = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/DiffusionNoWound1.npy',allow_pickle=True)
#EnsembleOut = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/DiffusionMutant1.npy',allow_pickle=True)
WTsampler = EnsembleOut[0]
outMut = WTsampler.get_chain(discard=500,thin = 25, flat=True)
"""
mutant = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/cont_delta.npy',allow_pickle=True)
Mutant= mutant[0]
outMut = Mutant.get_chain(discard=500,thin =5, flat=True)

"""
names = ['$q$ [mMol min$^{-1}$]', '$D$ [$\mu m^{2}$ min$^{-1}$]', 'τ [min]']#, '$R_0$ [mMol $\mu m^{-2}$]', '$\kappa_d$ [mMol $\mu m^{-2}$]', '$m$ [$\mu m^{2}$ mMol$^{-1}$]','b0']
names1 = ['q','D', 'τ']
priors  = [1000,1000,60]#,1,1,30,0.5]
col = ['r','b','g']
fig, axes = plt.subplots(nrows=3, ncols=1)
fig.suptitle("Mutant and unwounded:Inferred chemoattractant parameters",fontsize = 20)
sns.set_palette("muted")
for j in range(len(names)):
    axes[j].set_xlim(0,priors[j]) 

    axes[j].set_title(names[j],fontsize = 20)
    axes[j].set_yticks([])

   # sns.distplot(outMut[:, j], hist=False, kde_kws={"shade": False}, label = 'Mutant', ax=axes[j])]sns.distplot(outWT[:, j], hist=False,kde_kws={"shade": True},label = 'No wound', ax=axes[j])
    sns.distplot(outMut[:, j], hist=False,kde_kws={"shade": True}, label = 'Synthetic', ax=axes[j])
    sns.distplot(outWT[:, j], hist=False, kde_kws={"shade": True}, label = 'Mutant', ax=axes[j])
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
"""
def Bias_persistance(x,y):
    WT = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/WildTypeData-{}{}.npy'.format(x,y),allow_pickle=True)
    sampler = WT[0]
    WildTypeChain = sampler.get_chain(discard=250,thin = 2,flat=True)    
    W =  WildTypeChain[:,0]
    B =  WildTypeChain[:,2]
    OBWT = (W * B)
    return OBWT


def OBSIZE(val):
        OBWT = []
        for i in range(7):
            DataSet =  Bias_persistance(i,val)
            WT = DataSet
            OBWT.append(WT)
        return OBWT
# This plots a single time bin and all spatial bins for both mutant and WT ""
        


OB = OBSIZE(0)
MeanWT = np.mean(OB,axis=1)
stdWT = np.std(OB,axis=1)




fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True)
    
# instantiate a point wound
wound = PointWound(np.array([0,0]))


r_points = np.array([25, 50, 75, 100, 125, 150,175])


# where to measure observed bias
r1 = np.linspace(25, 95, 100)
r1 = np.linspace(25,200,100)
r2 = np.linspace(25, 95, 100)

r = np.linspace(25,180,100)
t = np.array([5,15,30,50])

for ax, p in zip(axes, t):
    ax.set_ylabel('$t={}$'.format(p), rotation=0, size='large', labelpad=35)
lines = []
scatters = []
for i, tt in enumerate(t):
    OB = OBSIZE(i)
    MeanWT = np.mean(OB,axis=1)
    stdWT = np.std(OB,axis=1)
    for theta in outMut[np.random.randint(len(outMut), size=100)]:
        col = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        q = theta[0]
        D = theta[1]

        r0,kd,m,b0 = theta[2],theta[3],theta[4],theta[5]
        params = [q, D, r0,kd,m,b0]
        ob1 = observed_bias(params, r, tt, wound)
        lines.append(axes[i].plot(r, ob1, color=col, linewidth=1,alpha=0.1)[0])
        axes[i].set_ylim(0, 0.3)
        axes[i].errorbar(r_points,MeanWT,yerr = stdWT,fmt='r.',capsize=5)

axes[0].set_title('Observed bias as a function of distance')
axes[-1].set_xlabel('Distance ($\\mu m$)')
plt.legend()
plt.tight_layout()



import corner
names = ['$q$', '$D$','$R_0$', '$\kappa_d$]', '$m$','b0']

fig = corner.corner(outMut, labels=names,quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": 12})
"""
bbox = dict(boxstyle="round", fc="0.8")
arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle,angleA=0,angleB=90,rad=10")

offset =  -50
axes[2].annotate('Experiment duration = {} mins'.format(50),
            (50,0), xytext=(5*offset, offset+ 80), textcoords='offset points',
            bbox=bbox, arrowprops=arrowprops,fontsize = 12)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])



mutant = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/DiffusionWildType.npy',allow_pickle=True)
Mutant= mutant[0]
outMut = Mutant.get_chain(discard=500,thin =5, flat=True)
names = ['$q$ [mMol min$^{-1}$]', '$D$ [$\mu m^{2}$ min$^{-1}$]', 'τ [min]']#, '$R_0$ [mMol $\mu m^{-2}$]', '$\kappa_d$ [mMol $\mu m^{-2}$]', '$m$ [$\mu m^{2}$ mMol$^{-1}$]','b0']
names1 = ['q','D', 'τ']
priors  = [10000,10000,60]#,1,1,30,0.5]
col = ['r','b','g']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(25, 5), sharex='col')
fig.suptitle("Wild Type:Inferred chemoattractant parameters",fontsize = 18)
sns.set_palette("muted")
for j in range(len(names)):
    axes[j].set_xlim(0,priors[j])
    axes[j].set_title(names[j],fontsize = 15)
    axes[j].set_yticks([])
    sns.distplot(outMut[:, j], hist=False, kde_kws={"shade": False}, label = 'Mutant', ax=axes[j])
    #sns.distplot(outWT[:, j], hist=False,kde_kws={"shade": False}, label = 'No wound', ax=axes[j])
   
bbox = dict(boxstyle="round", fc="0.8")
arrowprops = dict(
    arrowstyle = "->",
    connectionstyle = "angle,angleA=0,angleB=90,rad=10")

offset =  80
axes[2].annotate('Final time bin= {} mins'.format(45),
            (45,0), xytext=(-2*offset, offset), textcoords='offset points',
            bbox=bbox, arrowprops=arrowprops,fontsize = 12)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
"""