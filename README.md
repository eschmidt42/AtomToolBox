# AtomToolBox

AtomToolBox is a collection of Python codes I have developed during my PhD working with atomistic simulations using Molecular Dynamics ([LAMMPS](http://lammps.sandia.gov/), MD) and Density Functional Theory ([CASTEP](http://castep.org/), DFT). This package combines the ase and scikit-learn package to develop features of local atomic configurations to train classifiers and regress electron densities.

## Installation

Clone or download this repository. A proper installation is not set up yet. The requirements are listed in `requirements.txt`.

You may also need:

- for electron density regression: [pottytrainer](https://github.com/eschmidt42/PottyTrainer).
- for RVMs: [RVMs](https://github.com/eschmidt42/RVMs)

## Feature Generation

To train models for regression or prediction we require a list of features for each point of interest, e.g. atoms. In the notebook *"Ultracelling and featurizing.ipynb"* the pipeline of generating / parsing a crystal from disk, calculating required neighbourhood information and the subsequent features is demonstrated.

## Classification

For the classification of local atomic environments functions are implemented to read and write custom LAMMPS trajectories. The trajectories can then be used to generate classifiers or to be classified themselves. 

The focus of my PhD was to identify chemical ordering and track clustering and grain boundary migration in bicrystals in MD. A list of the corresponding notebooks / tutorials:

* Generating classifiers for unperturbed crystals and comparison: *Classification of an ideal crystals.ipynb*

* Create a single crystal and emulating LAMMPS simulations: *Create crystal and write LAMMPS traj file.ipynb*

* Create a bicrystal using GBpy and emulate LAMMPS simulations: *Creating GBs in fcc and Atom Classifiers.ipynb*

* Generation of a collection of perturbed single crystals and training + evaluation of various classifiers: *Classification of perturbed crystals.ipynb*

* Generation of a collection of perturbed single crystals and training + evaluation of a Gaussian Mixture Classifier: *Classification of perturbed Crystals using Gaussian Mixture Models.iypnb*

* Generation of a collection of perturbed single crystals ($\gamma$-phase, $\gamma^\prime$-phase and pure fcc Ni) and training of a Gaussian Mixture Classifier: *Classification of perturbed Crystals using Gaussian Mixture Models - gamma and gamma prime.iypnb*

* Generation of a collection of perturbed single crystals ($\gamma$-phase, $\gamma^\prime$-phase and pure fcc Ni) and training of a Gaussian Mixture Classifier using the decomposition algorithm to infer phases from convoluted feature density distributions: *Classification of perturbed Crystals using Gaussian Mixture Models and decomposition - gamma and gamma prime.ipynb*

* Analysis of MD simulations of single crystals in the $\gamma$-phase containing Al and Ni only: *Post-MD Analysis - Ordering and Precipitation.ipynb*

* Analysis of MD simulations of bicrystals containing Al and Ni only: *Post-MD Analysis - Tracking of GBs.iypnb*

## Electron Density Regression

In order to perform electron density regression on results from simulations with [Profess](http://cpc.cs.qub.ac.uk/summaries/AEBN_v3_0.html) or CASTEP use *"Electron density regression.ipynb"*. The regression returns 2 and 3-body functions. The former of which can be used for generation of Embedded Atom Method potentials. The last step is still ongoing work, but will be added to the AtomToolBox in the future.
