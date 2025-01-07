# Immune Cell Migration Analysis

The repository holds code for the following paper:

  Drosophila Mcr is a functional homologue of mammalian complement C5a and operates as a wound induced chemotactic signal to drive inflammatory cell recruitment to sites of sterile tissue damage
  
  Luigi Zechini1, Henry Todd1, Thibaut Sanchez1, Daniel R. Tudor1, Jennie S. Campbell1, Edward
 Antonian1, Stephen J. Jenkins1, Christopher D. Lucas1, Andrew J. Davidson2, Jean van den
 Elsen3, Linus J. Schumacher1*, Alessandro Scopelliti1* and Will Wood1,4*.

 1Institute for Regeneration and Repair, University of Edinburgh, Edinburgh BioQuarter, 4-5
 Little France Drive, Edinburgh EH16 4UU, UK
 2 Wolfson Wohl Cancer Research Centre, School of Cancer Sciences, University of Glasgow,
 Garscube Estate, Switchback Road, Bearsden, Glasgow, G61 1BD.
 3Department of Life Sciences, University of Bath, Claverton Down, Bath, UK, BA2 7AY
 4Lead contact
 *Correspondence: Linus.Schumacher@ed.ac.uk (L.J.S) alessandro.scopelliti@ed.ac.uk (A.S.)
 and w.wood@ed.ac.uk (W.W.)

This repo started out by trying to reproduce the computational methods in Weavers, Liepe, et al. "Systems Analysis of the Dynamic Inflammatory Response to Tissue Damage Reveals Spatiotemporal Properties of the Wound Attractant Gradient. Curr. Biol. 2016;26(15):1975–1989. http://dx.doi.org/10.1016/j.cub.2016.06.012, and is thus heavily based on their methodology, but now implemented in open-source Python code, with some added functionality. For introductory tutorials, please refer to the repository that this is forked from: https://github.com/nickelnine37/DrosophilaWoundAnalysis

The requirements for this project are Numpy, Pandas, Scipy, Matplotlib, Skimage, tqdm, emcee and Jupyter. It has been tested with the following versions:

```
Python version:     3.8.17
Numpy version:      1.20.3
Matplotlib version: 3.3.2
Pandas version:     0.25.3
Skimage version:    0.16.2
Scipy version:      1.3.3
tqdm version:       4.64.0
Jupyter version:    4.4.0
emcee version:    3.1.4
```

but *should* work with python 3 and any other set of compatible package versions.

Start in the Notebooks folder!
