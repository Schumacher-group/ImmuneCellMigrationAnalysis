# Immune Cell Migration Analysis

The repository holds code for analysing cell migration using a biased-persistent random walk as well as attractant dynamics.
This repo started out by trying to reproduce the computational methods in Weavers, Liepe, et al. "Systems Analysis of the Dynamic Inflammatory Response to Tissue Damage Reveals Spatiotemporal Properties of the Wound Attractant Gradient. Curr. Biol. 2016;26(15):1975–1989. http://dx.doi.org/10.1016/j.cub.2016.06.012. Here we expand upon that functionality. For a stable version, please refer to the repository that this is forked from: https://github.com/nickelnine37/DrosophilaWoundAnalysis

The requirements for this project are Numpy, Pandas, Scipy, Matplotlib, Skimage, tqdm and Jupyter. It has been tested with the following versions:

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

In addition, in order to input mp4 files, the module `ffmpeg-python` is needed. This can be installed by running

```
pip install ffmpeg-python
```

Start in the Notebooks folder!
