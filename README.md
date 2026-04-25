# Generalized Generative Quantum Eigensolver

This repository contains the codebase for an experimental research project applying Reinforcement Learning (RL) to Generative Quantum Eigensolvers (GQE). The goal of this project is to autonomously design quantum circuit ansätze capable of finding the ground state energy of complex quantum Hamiltonians using an autoregressive Transformer policy.

Experimentation Notes are recorded here: https://docs.google.com/document/d/17a2URHEqdNOEp_vzKegu2fZeEMuczSQDS9jdUm0Qczk/edit?usp=sharing

## The Problem

Variational Quantum Eigensolvers (VQE) typically rely on fixed, parameterized circuit architectures (ansätze) that are prone to issues like barren plateaus and local minima in complex energy landscapes (e.g., spin systems or molecular Hamiltonians). 

Recent innovations like the Generative Quantum Eigensolver (GQE) and SpinGQE successfully introduced autoregressive transformers to unconditionally generate quantum circuits. However, these models were trained using supervised learning, which requires pre-calculating target circuits or exact wavefunctions using classical heuristics. This creates a hard ceiling: **a quantum generator trained via supervised learning can never surpass the classical methods used to generate its training data.**

## Current Approach

This project removes the supervised learning bottleneck by utilizing a **pure Reinforcement Learning (RL) approach**. 

We treat quantum circuit generation as a Markov Decision Process (MDP). An autoregressive GPT model acts as the policy $\pi_\theta$, appending one quantum gate at a time to build a circuit topology. The model receives a reward signal based strictly on the energy expectation value $\langle \psi | H | \psi \rangle$ of the generated circuit.

**Key Technical Strategies:**
* **Policy Optimization:** Utilizes Group Relative Policy Optimization (GRPO) and Proximal Policy Optimization (PPO).
* **Self-Imitation Learning (SIL):** Mitigates catastrophic forgetting and sample inefficiency by maintaining an `EliteReplayBuffer` of high-reward circuit topologies to anchor the policy.
* **Dynamic Exploration:** Implements simulated annealing and dynamic temperature scaling to navigate out of local energy minima.
* **Action Masking:** Prevents redundant gate placements (e.g., consecutive identical Pauli operators on the same wires) to prune the search space.


## Repository Structure

The project is modularized into distinct domains handling the model, the physics engine, and the RL algorithms.

```text
├── config.py              # Centralized hyperparameters (RL, Generation, Physics)
├── train.py               # Main training orchestration and execution loop
├── requirements.txt       # Python requirements to install in venv
├── .gitignore             # Git Ignored files
├── README.md              # This guide
├── models/
│   ├── model.py           # Core Transformer/GPT architecture definitions
│   └── GPTQE.py           # Quantum-specific generation wrappers and masking logic
├── rl/
│   ├── losses.py          # Implementations for GRPO, PPO, and SIL loss functions
│   └── buffer.py          # EliteReplayBuffer for Self-Imitation Learning
├── physics/
│   ├── hamiltonian.py     # Hamiltonian matrix generation and definitions
│   └── evaluator.py       # Quantum simulation to calculate subsequence energies
└── utils/
    └── tracking.py        # Artifact saving, metric logging, and visualization tools
```


## Getting Started

Follow these steps to set up the environment and begin training.

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Set Up the Python Environment
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Import the Dataset
Hamiltonians and ground state labels are pulled from the open-source VQE Generated Dataset.
1. Clone the dataset repository:
   ```bash
   git clone https://github.com/Qulacs-Osaka/VQE-generated-dataset.git
   ```
2. Ensure the `VQE-generated-dataset` folder is located in the root directory of this project so `physics/hamiltonian.py` and `train.py` can correctly resolve the `.jb` file paths.

### 4. Configure and Train
1. Open `config.py` to adjust hyperparameters (e.g., number of qubits, RL algorithm choice, learning rate, buffer sizes).
2. Execute the training pipeline:
   ```bash
   python train.py
   ```
Artifacts, including loss curves, energy evaluation plots, and `.pt` model checkpoints, will be saved automatically to your designated output directory.


## Future Directions & Scientific Study

This repository serves as a foundational proof-of-concept (achieving ~96% accuracy on a 1D Isotropic Heisenberg XXX Hamiltonian). To transition this into a generalized solver, the following experiments and architectural upgrades are required:

### Immediate Experimentation (Short-Term)
* **Buffer Pre-filling for GRPO:** Currently, the initial buffer fill is gated behind the `ppo_sil` algorithm. Because GRPO also utilizes the SIL anchor, the `fill_initial_buffer()` method must be run unconditionally before training starts to prevent the SIL loss from pulling toward an empty state during early epochs.
* **Dynamic Buffer Floor Anchoring:** The initial absolute minimum floor for the SIL buffer is currently hardcoded to `-4.0`. For unknown Hamiltonians, this is a flawed assumption. The algorithm should be updated to run a preliminary generation batch (e.g., 100 random rollouts), evaluate the energies, and set the absolute floor to some top percentile (80th percentile or the top 20%) of that initial batch.

### Generalization & Encoding (Long-Term)
* **Hamiltonian Tokenization (Context-Awareness):** Currently, the model unconditionally generates circuits and over-fits to a single static Hamiltonian environment. To generalize the model, we must implement an Encoder network (e.g., a Graph Neural Network or a matrix tokenizer).
* **Multi-Task Training:** Once the encoder is implemented, the training pipeline should dynamically sample different Hamiltonians (varying $J$ coupling constants, magnetic field strengths, or molecule geometries) from the dataset during training. This will force the GPT to condition its generated circuit topology on the specific physical parameters provided by the encoder, effectively creating a generalized Quantum Eigensolver.