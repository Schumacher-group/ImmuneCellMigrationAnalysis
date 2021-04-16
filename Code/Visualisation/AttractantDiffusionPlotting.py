# allow imports from the LM package
import os
import sys
sys.path.append(os.path.abspath('..'))
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from in_silico.sources import CellsOnWoundMargin, CellsInsideWound, PointWound


# Set the Diffusion parameters
q, D, tau = 3196, 1692, 30

# Here we use rr but this is the same as the y-direction
rr = np.arange(0,(229 - 104),0.5)
# Time interval, we want to output a concentration every 10 minutes, from 5 minutes after production as started
t = np.arange(5,35,7)



# Location of the two wounds, and the size of the wounds. Here we assume two wounds of the same size.
# You could have wounds of varying sizes
cell_radius = 10
centre1 = np.array([0, (229 - 104)])
source1 = PointWound(position=centre1)
centre2 = np.array([0, 0])
source2 = PointWound(position=centre2)

# Plotting of the Attractant concentration at the different distances, instead of using source.concentration
# you can use source.concetration_xy to vary the spatial location in cartesian coordinates instead
for i in range(len(t)):
    conc1 = source1.concentration(np.array([q,D,tau]),rr, t[i])
    conc2 = source2.concentration(np.array([q,D,tau]),rr, t[i])
    reversed_conc2 = conc2[::-1]
    TotalConc= conc1  + reversed_conc2
    plt.plot((rr),TotalConc, label = "Time = {}mins".format(t[i]))
   #plt.plot((229 - rr),conc2, label = "Time = {}mins".format(t[i]))
    

#This plots the locations of the two wounds and size on the figure and shades them in
"""
plt.plot([219,219], [0, 1.6], 'k--', lw=2, label = "location of wound 1")
plt.plot([239, 239], [0, 1.6], 'k--', lw=2)
plt.plot([94, 94], [0, 1.6], 'k--', lw=2, label = "location of wound 2")
plt.plot([114, 114], [0, 1.6], 'k--', lw=2)

plt.axhspan(0,1.6,(107/511),(129/511),facecolor = "black",alpha=0.3)
plt.axhspan(0,1.6,(219/511),(245/511),facecolor = "black",alpha=0.3)
"""
plt.xlabel("radius")
plt.ylabel("Attractant concentration")

# Plot formatting and saving
plt.title("Attractant concentration through wounds")
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.show()
#plt.savefig("../Notebooks/AttractantConcentrationPW.pdf")
