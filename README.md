# Generative Pre-Trained Quantum Eigensolver

train.py ~ script to train a GPT-QE model. Hamiltonian, hyperparameters and model configuration can be freely changed.
GPT-QE.py ~ contains GPT-QE class code including forward pass, loss function calculation, and generation functions
model.py ~ contains nano GPT code that GPT-QE extends
hamiltonian.py ~ contains gen_hamiltonian function which generates hamiltonian based on inputted label (labels follow those in the VQE-generated-dataset)

Hamiltonians and ground states are pulled from the following dataset: https://github.com/Qulacs-Osaka/VQE-generated-dataset
make sure the /VQE-generated-dataset/data/ground_state/ is in your directory