from dataclasses import dataclass

@dataclass
class TrainConfig:
    # General
    seed: int = 1
    save_dir: str = "./experiments/annealing_longer_circ"
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

    # RL & Optimization
    n_epochs: int = 700
    n_batches: int = 4
    ppo_epochs: int = 4
    epsilon: float = 0.4
    lr: float = 1e-5
    weight_decay: float = 0.05
    entropy_coef: float = 0.05

    # Threshold Buffer Tracking
    buffer_size: int = 24
    buffer_floor: float = 4.0
    div_threshold: int = 2
    eval_iter: int = 1
    plot_iter: int = 50