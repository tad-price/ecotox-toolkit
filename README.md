# ecotox-toolkit

This repository contains Python (PyTorch/Numpy) implementations of models, datasets and evaluation functions that can be used for predicting the ecotoxicity of chemicals for aquatic species.

Implemented models: Factorization Machine, MLP, MLP+dim reduction, Cross network; Bayesian Factorization Machine with Blocked Gibbs Sampling

Experiments:
- Factorization machine for data gap filling A_0_FM_fill.py
- Predicting new chemical toxicities eg. A_1_m2v.py
- Filling data gaps with quantified uncertainty using the Bayesian FM
- Other small uncertainty demos in the notebook uncertainty.ipynb

This repository is built as part of a Master Thesis project at the University of Amsterdam and the Dutch National Institute of Public Health and Environment (RIVM). 
