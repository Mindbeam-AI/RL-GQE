from dataclasses import dataclass

@dataclass
class TrainConfig:
    # General
    seed: int = 1
    save_dir: str = "./experiments/ablation_studies/best2"
    load_model: bool = False
    
    # Quantum Environment
    # 1D Isotropic Heisenberg XXX Hamiltonian in external magnetic field (j=2), open BCs
    ham_label: int = 1 
    num_qubits: int = 4
    seq_len: int = 10

    # Generation
    seq_gen: int = 32
    gen_iter: int = 1
    temp_buffer_fill: float = 2.0
    temp_explore: float = 0.8
    temp_eval: float = 0.1

    # Dynamic Temperature Annealing
    temp_step: float = 0.05
    spike_magnitude: float = 0.5
    anneal_cutoff: float = 0.2
    temp_min: float = 1.0
    temp_max: float = 2.0
    stagnation_epochs = 10
    use_temp_decay: bool = False


    # RL & Optimization
    algo = "grpo" # Options: "grpo", "ppo_sil"
    beta: float = 0.02
    ref_sync_iter: int = 10
    sil_weight: float = 1.0
    sil_scaling: str = "linear" # Options: "linear", "exponential", "z-score"

    n_epochs: int = 300
    n_batches: int = 4
    ppo_epochs: int = 4
    epsilon: float = 0.4
    lr: float = 1e-5
    weight_decay: float = 0.05
    entropy_coef: float = 0.05

    # Threshold Buffer Tracking
    buffer_start_size: int = 24
    buffer_end_size: int = 4
    buffer_floor: float = 4.0
    div_threshold: int = 2
    
    eval_iter: int = 5
    plot_iter: int = 20

    # Discounted Future Rewards
    gamma: int = 0.8
    
    # Action Mask
    use_action_mask : bool = True