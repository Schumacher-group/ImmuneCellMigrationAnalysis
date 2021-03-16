

import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

niter = 7500
nwalkers = 200

Space = ["5 to 45 $\mu m$","25 to 75 $\mu m$","45 to 105 $\mu m$","65 to 135 $\mu m$","85 to 165 $\mu m$","105 to 195 $\mu m$"]
Time = ["Time = 5 mins","Time = 15 mins","Time = 30 mins","Time =  50 mins","Time =  80 mins","Time =  120 mins"]#,"Time =  40 mins","Time =  50 mins"]


# Load the posterior numpy arrays
def loadData(i,j):
    WT = np.load('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/TwoWoundControlloc3{}{}.npy'.format(i,j),allow_pickle=True)
    samplerWT = WT[0]
    WildTypeChain = samplerWT.get_chain(discard=250,thin = 2,flat=True)
    return WildTypeChain
#dataWT = loadData()
"""
WWT = dataWT[:,0]
PWT = dataWT[:,1]
BWT = dataWT[:,2]
OBWT = WWT * BWT
plt.hist(OBWT,label='$B$ = {:.2f} $\pm$ {:.2f}'.format(np.mean(OBWT), np.std(OBWT)),bins=100,alpha=0.6,density=True)
plt.savefig('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/Images/PosterImages/WalkerPoster/baseline_bias_total.pdf')


fig, ax = plt.subplots(6,4,figsize=(10,10),sharex=True)
fig.suptitle("Two wound - wound 3")
for i in range(6):
    for j in range(4):
        data = loadData(i,j)
        WWT = data[:,0]
        PWT = data[:,1]
        BWT = data[:,2]
        ax[i,j].set_ylim([0, 40])
        ax[i,j].hist(WWT,label='$WWT$ = {:.2f} $\pm$ {:.2f}'.format(np.mean(WWT), np.std(WWT)),bins=100,alpha=0.6,density=True)
        ax[i,j].hist(PWT,label='$PWT$ = {:.2f} $\pm$ {:.2f}'.format(np.mean(PWT), np.std(PWT)),bins=100,alpha=0.6,density=True)
        ax[i,j].hist(BWT,label='$BWT$ = {:.2f} $\pm$ {:.2f}'.format(np.mean(BWT), np.std(BWT)),bins=100,alpha=0.6,density=True)

        ax[0,j].set_title(Time[j])
        ax[i,0].set_ylabel(Space[i], rotation=90, size='small')
        ax[i,j].legend(prop={'size': 6})
plt.savefig('/Users/danieltudor/Documents/ImmuneCellMigrationAnalysis/Images/PosterImages/WalkerPoster/Two-wound-loc3.pdf')
"""

#plt.show()
"""
## This outputs the trace for a specified np.array
fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True)
for i in [0,1,2]:
    dist_out = loadData(5,0)
    axes[i].plot(dist_out[:, i], linewidth=0.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i], alpha=0.7)
    axes[i].set_title(['$w$', '$p$', '$b$'][i])

axes[1].set_ylabel('Sampled value', size='large')
axes[2].set_xlabel('MCMC step', size='large')
plt.tight_layout()

'../data/np_array/Two Wound Control New Params-{}{}'
"""

def Bias_persistance(x,y):
    WT1 = np.load('/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/TwoWoundControlloc1{}{}_new1.npy'.format(x,y),allow_pickle=True)
    WT2 = np.load('/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/TwoWoundControlloc2{}{}_new.npy'.format(x,y),allow_pickle=True)
    WT3 = np.load('/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/TwoWoundControlloc3{}{}_new1.npy'.format(x,y),allow_pickle=True)

    samplerWT1 = WT1[0]
    WildTypeChain1 = samplerWT1.get_chain(discard=250,thin = 2,flat=True)
    samplerWT2 = WT2[0]
    WildTypeChain2 = samplerWT2.get_chain(discard=250,thin = 2,flat=True)
    samplerWT3 = WT3[0]
    WildTypeChain3 = samplerWT3.get_chain(discard=250,thin = 2,flat=True)
    W1 =  WildTypeChain1[:,0]
    B1 =  WildTypeChain1[:,2]
    OBWT1 = (W1 * B1)
    W2 =  WildTypeChain2[:,0]
    B2 =  WildTypeChain2[:,2]
    OBWT2 = (W2 * B2)
    W3 =  WildTypeChain3[:,0]
    B3 =  WildTypeChain3[:,2]
    OBWT1 = (W1 * B1)
    OBWT2 = (W2 * B2)
    OBWT3 = (W3 * B3)




    return OBWT1,OBWT2,OBWT3


def OBSIZE(val):
        OBWT1 = []
        OBWT2 = []
        OBWT3 = []

        for i in range(7):
            WT1 =  Bias_persistance(i,val)[0]
            WT2 =  Bias_persistance(i,val)[1]
            WT3 =  Bias_persistance(i,val)[2]

            OBWT1.append(WT1)
            OBWT2.append(WT2)
            OBWT3.append(WT3)

        return OBWT1,OBWT2,OBWT3
# This plots a single time bin and all spatial bins for both mutant and WT ""

fig, ax = plt.subplots(4,1,figsize=(5,5),sharex=True)
colors = ['k', 'r','b']
Time = ["0-10","5-25","15-45","35-65"]
fig.suptitle("Two Wound Observed Bias - Control",fontsize=16)

for i in range(4):
    OB = OBSIZE(i)
    OBWT1= OB[0]
    OBWT2 = OB[1]
    OBWT3 = OB[2]
    MeanWT1 = np.mean(OBWT1,axis=1)
    stdWT1 = np.std(OBWT1,axis=1)
    MeanWT2 = np.mean(OBWT2,axis=1)
    stdWT2 = np.std(OBWT2,axis=1)
    MeanWT3 = np.mean(OBWT3,axis=1)
    stdWT3 = np.std(OBWT3,axis=1)

    x = [25,50,75,100,125,150,175]
    ax[i].errorbar(x,MeanWT2,yerr = stdWT2,fmt='o--',capsize=2,color = colors[1], label =  "Wound 1")
    ax[i].errorbar(x,MeanWT1,yerr = stdWT1,fmt='o--',capsize=2,color = colors[0], label =  "Wound 2")
    ax[i].errorbar(x,MeanWT3,yerr = stdWT3,fmt='o--',capsize=2,color = colors[2], label =  "Midpoint")

    #ax[i].errorbar(x,MeanMut,yerr = stdMut,fmt='o-',capsize=2,color = colors[i], label = "Mutant")
    #ax[i].errorbar(x,MeanMT,yerr = stdMT,fmt='o--',capsize=2,color = colors[1], label =  "Mutant")

    ax[i].set_ylabel(Time[i], rotation=90, fontsize = 10)
    ax[i].set_ylim(0,0.5)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.text(0.02,0.5, "Time bins (minutes)", ha="center", va="center", fontsize=12,rotation=90)
plt.legend()
plt.xlabel("Distance $\mu$m", fontsize=15)
plt.savefig("/Users/danieltudor/Desktop/Two_wound_control_new.pdf")
