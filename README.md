# RL-GQE: Reinforcement Learning towards Generative Quantum Eigensolving

[![Technical Breakdown](https://img.shields.io/badge/📖_Blog-Technical_Breakdown-blueviolet.svg)](https://mindbeam-website-git-rl-gqe-blog-mindbeam.vercel.app/research/rl-gqe)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository implements a **pure Reinforcement Learning (RL) approach** to Generative Quantum Eigensolving. It utilizes an autoregressive GPT model acting as a policy ($\pi_\theta$) to autonomously generate quantum circuits for highly frustrated spin systems. By coupling Group Relative Policy Optimization (GRPO) with Self-Imitation Learning (SIL), the agent successfully navigates complex barren plateaus without relying on classical heuristics or pre-computed optimal circuits.

**Strict No Data Leakage Guarantee:** While classical VQE datasets are included in the repository, they are used exclusively for post-run evaluation and benchmarking. The transformer policy is entirely self-taught, deriving its gradients solely from the physical energy evaluations ($\langle \psi \vert{} H \vert{} \psi \rangle$) of its self-generated circuits.

**Key Features:**
* **Policy Optimization:** Utilizes Group Relative Policy Optimization (GRPO) and Proximal Policy Optimization (PPO).
* **Self-Imitation Learning (SIL):** Mitigates catastrophic forgetting and sample inefficiency by maintaining an `EliteReplayBuffer` of high-reward circuit topologies to anchor the policy.
* **Dynamic Exploration:** Implements simulated annealing and dynamic temperature scaling to navigate out of local energy minima.
* **Action Masking:** Prevents redundant gate placements (e.g., consecutive identical Pauli operators on the same wires) to prune the search space.

## Repository Structure

The project is modularized into distinct domains handling the model architecture, the physics engine, and the RL training loop.

```text
├── config.py               # Centralized hyperparameters (RL, Generation, Physics)
├── train.py                # Main training orchestration and execution loop
├── requirements.txt        # Python requirements to install in venv
├── .gitignore              # Git ignored files
├── README.md               # This file
├── models/
│   ├── model.py            # Core transformer/GPT architecture
│   └── GPTQE.py            # Quantum-specific generation wrappers and masking logic
├── rl/
│   ├── losses.py           # Implementations for GRPO, PPO, and SIL loss functions
│   └── buffer.py           # EliteReplayBuffer for Self-Imitation Learning
├── physics/
│   ├── hamiltonian.py      # Hamiltonian matrix generation and definitions
│   └── evaluator.py        # Quantum simulation to calculate subsequence energies
└── utils/
    ├── tracking.py         # Artifact saving, metric logging, and visualization tools
    └── report_plotting.py  # Plot creation

```

## File Descriptions

**Root Directory**
* `config.py`: A centralized dataclass containing all hyperparameters for the quantum environment, GPT generation, RL algorithms, and temperature annealing schedules. 
* `train.py`: The main orchestration script that executes the training loop. It handles the dynamic environment setup, autoregressive circuit generation, reward calculation, and policy updates.

**`/models` (Neural Network Architecture)**
* `model.py`: Contains the core Transformer/GPT architecture used as the policy network ($\pi_\theta$), defining the self-attention blocks and embedding layers.
* `GPTQE.py`: Wraps the base GPT model with quantum-specific generation logic. It handles the autoregressive decoding step, token sampling, and the application of the logical action mask.

**`/rl` (Reinforcement Learning Mechanics)**
* `losses.py`: Implements the mathematical objective functions for the RL algorithms. This includes the calculations for Group Relative Policy Optimization (GRPO), Proximal Policy Optimization (PPO), and the Self-Imitation Learning (SIL) loss.
* `buffer.py`: Defines the `EliteReplayBuffer` used for Self-Imitation Learning. It dynamically tracks, sorts, and stores the highest-reward circuit trajectories to anchor the policy and prevent catastrophic forgetting.

**`/physics` (Quantum Environment & Simulation)**
* `hamiltonian.py`: Defines the quantum physics environment and the available operator pool. It constructs the target Hamiltonian matrices (e.g., Heisenberg XXX) based on the configurations loaded from the dataset.
* `evaluator.py`: Acts as the quantum simulator that calculates the expectation values ($\langle \psi \vert{} H \vert{} \psi \rangle$) for generated circuits. It maps discrete gate tokens to physical matrices to compute the step-by-step subsequence energies used for RL rewards.

**`/utils` (Logging & Visualization)**
* `tracking.py`: Manages the logging of training metrics, terminal print formatting, and the saving of `.pt` model checkpoints and artifact dictionaries.
* `report_plotting.py`: Contains visualization tools used to generate loss curves, energy convergence plots, and evaluation histograms from the saved training runs.


## Getting Started

Follow these steps to set up the environment and begin training.

### 1. Clone the Repository

```bash
git clone https://Mindbeam-AI/RL-GQE.git
cd https://Mindbeam-AI/RL-GQE.git

```

### 2. Set Up the Python Environment

It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

```

### 3. Import the Evaluation Dataset

Hamiltonians and ground state energies are pulled from the open-source VQE Generated Dataset strictly for evaluation plotting.

1. Clone the dataset repository:
```bash
git clone [https://github.com/Qulacs-Osaka/VQE-generated-dataset.git](https://github.com/Qulacs-Osaka/VQE-generated-dataset.git)

```

2. Ensure the `VQE-generated-dataset` folder is located in the root directory of this project so `physics/hamiltonian.py` and `train.py` can correctly resolve the `.jb` file paths.

### 4. Configure and Train

Execute the main training pipeline:

```bash
python train.py

```

Artifacts, including loss curves, energy evaluation plots, and `.pt` model checkpoints, will be saved automatically to the designated output directory.

## Configuration & Hyperparameters

All model and environment hyperparameters are centralized in `config.py` and can be changed before running `train.py`. The most common adjustable configurations and hyperparameters in `config.py` include:

```text
    # Quantum Environment
    num_qubits: int = 4         # System size, increases compute exponentially.
    ham_label: int = 1          # Selects which Hamiltonian to load from the VQE dataset.

    # Generation
    seq_gen: int = 32              # The base number of quantum circuits (trajectories) generated per rollout batch.
    gen_iter: int = 1              # Frequency multiplier for rollouts. Generation occurs every N epochs, scaling the total sequences generated.
    temp_buffer_fill: float = 2.0  # High sampling temperature used strictly for the initial random walk to seed the SIL buffer with diverse trajectories.
    temp_explore: float = 0.8      # Baseline sampling temperature during standard RL training to encourage exploration.
    temp_eval: float = 0.1         # Near-zero temperature used during evaluation to force greedy, high-confidence circuit generation without exploration noise.

    # Dynamic Temperature Annealing
    # Controls the temperature schedule for sequence generation to prevent local minima trapping.
    temp_min: float = 1.0       # Base temperature for exploitation.
    temp_max: float = 2.0       # Maximum temperature spike for exploration.
    spike_magnitude: float = 0.5# Magnitude of temperature spike applied upon stagnation.
    stagnation_epochs: int = 10 # Number of epochs stuck in a local minimum before a spike is triggered.

    # RL & Optimization
    algo: str = "grpo"          # Options: "grpo" (Group Relative Policy Optimization) or "ppo_sil".
    beta: float = 0.02          # KL divergence penalty coefficient for GRPO.
    sil_weight: float = 1.0     # Weight of the Self-Imitation Learning loss. Change to 0.0 to remove SIL from loss.
    n_epochs: int = 300         # Total number of generation and optimization loops.

    # Threshold Buffer Tracking
    # Dynamic capacity controls to force the model to exploit narrow success paths.
    buffer_start_size: int = 24 # Initial capacity for broad exploration.
    buffer_end_size: int = 4    # Final capacity to lock in the exact ground state.
    div_threshold: int = 2      # Threshold used to track and enforce sequence diversity within the buffer.

    # Discounted Future Rewards
    gamma: float = 0.8          # Discount factor weighing the value of final gate sequence energies. < 1.0 ensures early gates are optimized for the final sequence energy they lead to.

    # Action Mask
    use_action_mask : bool = True  # Enables logical pruning during generation (e.g., prevents placing consecutive identical gates on the same wire to avoid identity loops).

    # Logging & Evaluation
    eval_iter: int = 5          # Runs a deterministic evaluation step without exploration noise every N epochs.
    plot_iter: int = 20         # Generates and saves evaluation histograms every N epochs.
```

The config is already loaded to recreate the best model results for the 4 qubit 1D Isotropic Heisenberg XXX Hamiltonian with $h=2$. In `metadata.json` the ``best_eval_min`` is $-7.5173$ and the Hamiltonian ``ground_energy`` is $-7.8284$, constituting $96.03$% accuracy.

## Output Files

Upon completion of a training run, the pipeline automatically generates and saves several artifacts in the directory specified by `cfg.save_dir` (default: `./experiments/best`). These files provide everything needed to evaluate the model's physics accuracy and monitor training.

**Model Checkpoints & Data**
* `final.pt`: The PyTorch model checkpoint. Contains the finalized state dictionaries for the transformer model and the optimizer, allowing you to reload the exact policy network for future inference or transfer learning.
* `evals_E.csv`: The raw energy evaluation data. Logs the exact ground state expectation values ($\langle \psi \vert{} H \vert{} \psi \rangle$) of the circuits generated during the evaluation steps, stripping away exploration noise to show the true policy capability.
* `training_dynamics.csv`: A raw, epoch-by-epoch data log tracking training loss, KL divergence, active temperature, and the SIL buffer minimum energy. Useful for custom plotting.
* `metadata.json`: Stores the complete run configuration alongside the final performance metrics (e.g., `best_eval_min` and the true `ground_energy`).

**Visualizations & Plots**
* `diagnostics_fig.png`: A dashboard showing the dynamic temperature schedule and the `buffer_min` over time, proving the effectiveness of the Self-Imitation Learning (SIL) anchor against thermal runaway.
* `eval_fig.png`: The primary physics convergence plot. Maps the mean policy energy against the training epochs, including standard deviation shading and a horizontal reference line for the true ground state.
* `loss_fig.png`: The objective loss curve representing the explorer loss (GRPO/PPO) combined with the SIL penalty over the training duration. 
* `histo/histogram_epoch_[X].png`: Distribution plots generated during specific evaluation epochs (set by `plot_iter`), showing the actual spread of generated circuit energies compared to the ground truth.

## Related Work

This implementation builds upon:
- **SpinGQE**: [https://github.com/Mindbeam-AI/SpinGQE](https://github.com/Mindbeam-AI/SpinGQE)
- **PennyLane GQE Tutorial**: [https://pennylane.ai/qml/demos/gqe_training](https://pennylane.ai/qml/demos/gqe_training)
- **VQE Ground State Dataset**: [https://github.com/Qulacs-Osaka/VQE-generated-dataset](https://github.com/Qulacs-Osaka/VQE-generated-dataset)
- **nano-GPT**: Transformer architecture base

## Contact

This project was developed by Advaith Cheruvu under Mindbeam Technologies Inc. For questions or issues:
- **Email**: research@mindbeam.ai or advaith.cheruvu@mindbeam.ai
- **Issues**: [GitHub Issues](https://github.com/Mindbeam-AI/RL-GQE/issues)

## Acknowledgments

We thank the authors of the VQE ground state dataset for making their data publicly available.

---

**Mindbeam AI** | [Website](https://mindbeam.ai)
