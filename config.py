from dataclasses import dataclass

@dataclass
class TrainConfig:
    # General
    seed: int = 3
    save_dir: str = "./experiments/GRPO_lin1.0SIL_highDiscount_fastAnneal_Mask3"
    load_model: bool = False
    
    # Quantum Environment
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


    # RL & Optimization
    algo = "grpo" # can change to ppo_sil
    beta: float = 0.02
    ref_sync_iter: int = 10
    sil_weight: float = 1.0 # changed from 1.8, since SIL is linearized

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
    gamma = 0.80