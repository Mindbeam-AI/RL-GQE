# Generalized Generative Quantum Eigensolver

This repository implements a Reinforcement Learning (RL) framework to train a generative transformer (GPTQE) to autonomously construct quantum gate sequences that minimize the energy of a specific Hamiltonian. The system uses an Expert Iteration strategy using off-policy Self-Imitation Learning (SIL) for exploiting verified energy minimums.


Hamiltonians and ground states are pulled from the following dataset: https://github.com/Qulacs-Osaka/VQE-generated-dataset

make sure the /VQE-generated-dataset/data/ground_state/ is in your directory

Ensure your ground truth .jb files are correctly placed in the VQE-generated-dataset/ directory. Configure your hyperparameters in config.py, then initiate the training sequence:


python train.py