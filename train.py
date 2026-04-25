import os
import random
import numpy as np
import torch
import copy
import joblib

# Modular Imports
from config import TrainConfig
from physics import gen_hamiltonian, build_operator_pool, get_subsequence_energies
from rl import EliteReplayBuffer, compute_ppo_loss, compute_sil_loss, compute_grpo_loss
from models.model import GPTConfig
from models.GPTQE import GPTQE
from utils.tracking import plot_epoch_histogram, save_training_artifacts, print_training_step, print_eval_step

class RLTrainer:
    """
    Main orchestration class for training the Generative Quantum Eigensolver via RL.
    Manages environment setup, rollout generation, advantage estimation, and policy optimization.
    """
    
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.setup_env()
        self.setup_physics()
        self.setup_models()
        
        self.buffer = EliteReplayBuffer(max_size=cfg.buffer_start_size, absolute_floor=cfg.buffer_floor)
            
        # Tracking dictionaries
        self.history = {
            'losses': [], 'eval_Es': [], 'eval_epochs': [], 
            'kl_divs': [], 'active_temps': [], 'buffer_mins': []
        }
        self.best_min = float('inf')
        self.best_eval_epoch = 0

    def setup_env(self):
        """Seeds all random number generators for reproducibility."""
        
        os.environ['PYTHONHASHSEED'] = str(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def setup_physics(self):
        """Loads target Hamiltonian, true ground state, and initializes the operator pool."""
        
        data = joblib.load(f'./VQE-generated-dataset/data/ground_state/0{self.cfg.num_qubits}qubit/label{self.cfg.ham_label}.jb')
        self.grd_E = data["ground_energy"]
        self.ham = gen_hamiltonian(self.cfg.ham_label, self.cfg.num_qubits)
        self.init_state = [0] * self.cfg.num_qubits
        self.op_pool = np.array(build_operator_pool(self.cfg.num_qubits), dtype=object)

        # Baseline energy before any gates are applied (empty circuit)
        empty_seq = np.array([], dtype=object)
        base_E_array = get_subsequence_energies(empty_seq, self.ham, self.init_state, self.cfg.num_qubits)
        self.base_E = base_E_array[-1] if len(base_E_array) > 0 else 0.0

        # Pre-compute valid action mask (Rule: No consecutive identical operations on same wires)
        vocab_size = len(self.op_pool) + 1
        self.action_mask = torch.zeros((vocab_size, vocab_size), dtype=torch.bool, device="cuda")

        for i, op1 in enumerate(self.op_pool):
            idx1 = i + 1  # Offset by 1 because 0 is the start token
            for j, op2 in enumerate(self.op_pool):
                idx2 = j + 1
                # Apply Action Mask Rule
                if op1.name == op2.name and op1.wires == op2.wires:
                    self.action_mask[idx1, idx2] = True

    def setup_models(self):
        """Initializes the active GPT model, the optimizer, and the frozen reference model."""
        
        gpt_cfg = GPTConfig(
            vocab_size=len(self.op_pool) + 1, block_size=self.cfg.seq_len, dropout=0.0,
            bias=False, n_layer=4, n_embd=384, n_head=8
        )
        self.model = GPTQE(gpt_cfg).to("cuda")
        self.opt = self.model.configure_optimizers(
            weight_decay=self.cfg.weight_decay, learning_rate=self.cfg.lr, betas=(0.9, 0.95), device_type="auto"
        )
        
        if self.cfg.load_model:
            load = torch.load("./saved_models/final.pt", map_location="cpu")
            self.model.load_state_dict(load["model_state_dict"])
            self.opt.load_state_dict(load["optimizer_state_dict"])
            
        self.old_model = copy.deepcopy(self.model)
        for p in self.old_model.parameters(): p.requires_grad = False
        self.old_model.eval()

    def generate_rollouts(self, epoch, temp=None):
        """Executes the environment interaction phase to gather trajectories, rewards, and advantages."""
        
        current_temp = temp if temp is not None else self.cfg.temp_explore
        self.model.eval()

        # PHASE 1: AUTOREGRESSIVE GENERATION
        
        tokens, _ = self.model.generate(
            n_sequences=self.cfg.seq_gen * self.cfg.gen_iter,
            max_new_tokens=self.cfg.seq_len, 
            temperature=current_temp,
            device="cuda",
            action_mask=self.action_mask if self.cfg.use_action_mask else None # apply action mask
        )

        # PHASE 2: REWARD CALCULATION (PHYSICS)

        # Map tokens back to physics operators and evaluate expectation values
        gen_op_seq = self.op_pool[(tokens[:, 1:] - 1).cpu().numpy()]
        energies = torch.tensor(
            get_subsequence_energies(gen_op_seq, self.ham, self.init_state, self.cfg.num_qubits),
            device="cuda", dtype=torch.float32
        )
        
        rewards = -energies[:, -1]

        # PHASE 3: ALGORITHM ROUTING & ADVANTAGES
        
        if self.cfg.algo == "grpo":
            
            # 3A. Calculate Step Rewards (Energy Change - Gate Penalty)
            init_E_tensor = torch.full((energies.size(0), 1), self.base_E, device="cuda", dtype=torch.float32)
            full_E_trajectory = torch.cat([init_E_tensor, energies], dim=1)
            
            # 1. Calculate immediate step rewards
            energy_change = full_E_trajectory[:, :-1] - full_E_trajectory[:, 1:]
            gate_cost = 0.005 
            step_rewards = energy_change - gate_cost
            
            # 3B. Calculate Discounted Future Returns
            # This ensures early gates are judged by the final sequence energy they lead to.
            returns = torch.zeros_like(step_rewards)
            returns[:, -1] = step_rewards[:, -1]

            for t in reversed(range(step_rewards.size(1) - 1)):
                returns[:, t] = step_rewards[:, t] + self.cfg.gamma * returns[:, t+1]
            
            # 3C. Group Normalization
            mean_ret = returns.mean(dim=0, keepdim=True)
            std_ret = returns.std(dim=0, keepdim=True) + 1e-8
            
            final_advs = (returns - mean_ret) / std_ret

            if epoch % self.cfg.ref_sync_iter == 0:
                self.old_model.load_state_dict(self.model.state_dict())
            
            # 3D. SIL Buffer Insertion
            global_baseline = self.buffer.rewards[-1].item() if (self.buffer is not None and self.buffer.is_full()) else self.cfg.buffer_floor
            buffer_mask = rewards > global_baseline
            if buffer_mask.sum() > 0:
                dummy_advs = torch.ones_like(rewards) 
                self.buffer.add(tokens, rewards, dummy_advs)
                
            
        elif self.cfg.algo == "ppo_sil":
            global_baseline = self.buffer.rewards[-1].item() if self.buffer.is_full() else self.cfg.buffer_floor
            seq_advs = torch.zeros_like(rewards)
            mask = rewards > global_baseline
            
            if mask.sum() > 0:
                seq_advs[mask] = torch.clamp(rewards[mask] - global_baseline, min=0.1, max=3.0)
            
            self.buffer.add(tokens, rewards, seq_advs)
            self.old_model.load_state_dict(self.model.state_dict())
            
            # Expand 1D advantage to match token sequence shape
            final_advs = seq_advs.unsqueeze(-1).expand_as(tokens[:, :-1])
            
        return tokens, final_advs, rewards

    def update_policy(self, tokens, advantages, epoch, rewards):
        """Applies PPO/GRPO and SIL losses to update the policy network."""
        self.model.train()
        loss_record = 0.0
        kl_record = 0.0 
        
        for _ in range(self.cfg.ppo_epochs):
            self.opt.zero_grad()
            loss_sil = torch.tensor(0.0, device="cuda")

            # STEP 1: SIL ANCHOR LOSS
            if self.buffer is not None and getattr(self.buffer, 'tokens', None) is not None:
                dynamic_floor = self.buffer.rewards[len(self.buffer.rewards)//2].item() if self.buffer.is_full() else self.cfg.buffer_floor
                sil_toks, sil_rews = self.buffer.sample(self.cfg.seq_gen)
                if sil_toks is not None:
                    loss_sil = compute_sil_loss(
                        self.model, sil_toks, sil_rews, 
                        floor=dynamic_floor, scaling=self.cfg.sil_scaling
                    )

            # STEP 2: EXPLORER LOSS (GRPO OR PPO)
            if self.cfg.algo == "grpo":
                loss_explorer, kl_mean = compute_grpo_loss(
                    self.model, tokens, advantages, self.cfg.epsilon, self.old_model, self.cfg.beta
                )
                kl_record += kl_mean.item()
                loss = loss_explorer + (self.cfg.sil_weight * loss_sil)

            elif self.cfg.algo == "ppo_sil":
                loss_explorer = compute_ppo_loss(
                    self.model, tokens, advantages, self.cfg.epsilon, self.old_model, self.cfg.entropy_coef
                )
                loss = loss_explorer + (self.cfg.sil_weight * loss_sil)
            
            # STEP 3: BACKPROPAGATION
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                loss_record += loss.item()
                    
        avg_loss = loss_record / self.cfg.ppo_epochs
        avg_kl = (kl_record / self.cfg.ppo_epochs) if self.cfg.algo == "grpo" else 0.0
        return avg_loss, avg_kl

    def evaluate(self, epoch):
        """Deterministic evaluation step to verify true policy improvement without exploration noise."""
        
        self.model.eval()
        tokens, _ = self.model.generate(
            n_sequences=10, 
            max_new_tokens=self.cfg.seq_len, 
            temperature=self.cfg.temp_eval, 
            device="cuda", 
            action_mask=self.action_mask if self.cfg.use_action_mask else None # apply action mask
        )

        eval_op_seq = self.op_pool[(tokens[:, 1:] - 1).cpu().numpy()]
        true_Es = get_subsequence_energies(eval_op_seq, self.ham, self.init_state, self.cfg.num_qubits)[:, -1].reshape(-1, 1)
        
        eval_mean = np.mean(true_Es)
        eval_min = np.min(true_Es)
        unique_cnt = torch.unique(tokens, dim=0).size(0)

        self.history['eval_Es'].append(true_Es)
        self.history['eval_epochs'].append(epoch)

        # Call tracking function
        print_eval_step(
            epoch=epoch,
            eval_min=eval_min,
            eval_mean=eval_mean,
            unique_cnt=unique_cnt,
            total_seqs=len(tokens),
            temp_eval=self.cfg.temp_eval,
            eval_seq=tokens[0, 1:].cpu().numpy()
        )
        
        if eval_min < self.best_min:
            self.best_min = eval_min
            self.best_eval_epoch = epoch

        if epoch % self.cfg.plot_iter == 0:
            plot_epoch_histogram(true_Es.flatten(), epoch, self.grd_E, self.cfg.save_dir)

    def fill_initial_buffer(self):
        """Pre-fills the SIL buffer with random exploration before structured PPO training begins."""
    
        print("\n--- Filling Initial Buffer ---")
        fill_step = 1
        while not self.buffer.is_full():
            self.generate_rollouts(epoch=0)
            b_size, b_mean, b_max = self.buffer.stats()
            print(f"Fill Step {fill_step} | Buffer: {b_size}/{self.cfg.buffer_start_size} | Mean E: {-b_mean:.4f} | Best E: {-b_max:.4f}")
            fill_step += 1
        print("--- Buffer Full. Starting Training ---\n")

    def train(self):
        """Main training loop orchestrating generation, environment dynamics, and policy updates."""
        
        if self.cfg.algo == "ppo_sil":
            self.fill_initial_buffer()
        
        step_temp = 0.0
        spike_temp = 0.0
        stagnation_counter = 0
        best_window_min = float('inf')
        
        for epoch in range(self.cfg.n_epochs + 1):
            active_temp = min(self.cfg.temp_min + step_temp + spike_temp, self.cfg.temp_max)

            # STEP 1: DYNAMIC BUFFER SIZE MANAGEMENT
            # Stay at Max for the first 25% of training
            burn_in_threshold = int(self.cfg.n_epochs * 0.25)
            
            if epoch < burn_in_threshold:
                current_buffer_size = self.cfg.buffer_start_size
            else:
                # Exponentially decay for the remaining 75%
                t_rel = epoch - burn_in_threshold
                t_rem = self.cfg.n_epochs - burn_in_threshold
                
                # Calculate the decay rate constant k
                ratio = self.cfg.buffer_end_size / self.cfg.buffer_start_size
                k = -np.log(ratio) / t_rem
                
                # Calculate exponential size and round to nearest integer
                current_buffer_size = int(self.cfg.buffer_start_size * np.exp(-k * t_rel * 2))
                
                # Ensure we never drop below the minimum floor
                current_buffer_size = max(current_buffer_size, self.cfg.buffer_end_size)

            self.buffer.shrink_capacity(current_buffer_size)
            
            # STEP 2: GENERATION & DYNAMIC TEMPERATURE SCHEDULE
            
            if epoch % self.cfg.gen_iter == 0:
                tokens, advantages, rewards = self.generate_rollouts(epoch=epoch, temp=active_temp)
                
                gen_energies = -rewards
                gen_mean = gen_energies.mean().item()
                gen_min = gen_energies.min().item()
                gen_std = gen_energies.std().item() # NEW: Calculate standard deviation
                unique_cnt = torch.unique(tokens, dim=0).size(0)
                
                spike_temp = 0.0 
                # Check if better minimum found
                if gen_min < best_window_min - 0.02: 
                    best_window_min = gen_min
                    stagnation_counter = 0
                    
                    # Cool down: exploit valley
                    if step_temp > 0.0:
                        step_temp = max(step_temp - self.cfg.temp_step * 2, 0.0)
                else:
                    stagnation_counter += 1

                # If stuck in the same energy band for stagnation epochs
                if stagnation_counter >= self.cfg.stagnation_epochs:

                    if self.cfg.use_temp_decay:
                        # Spikes get exponentially weaker as training progresses
                        decay_ratio = max(0.0, 1.0 - (epoch / self.cfg.n_epochs))
                    else:
                        # Constant spikes throughout training
                        decay_ratio = 1.0
                    
                    if decay_ratio < 0.15:
                        spike_temp = 0.0
                        step_temp = 0.0
                    else:
                        spike_temp = self.cfg.spike_magnitude * decay_ratio
                        step_temp = min(step_temp + (self.cfg.temp_step * decay_ratio), self.cfg.temp_max - self.cfg.temp_min)
                    
                    stagnation_counter = 0 # Reset the counter
                    best_window_min = gen_min # Reset the baseline
                    

            # STEP 3: OPTIMIZATION
            
            avg_loss, avg_kl = self.update_policy(tokens, advantages, epoch, rewards)
            
            # STEP 4: LOGGING & EVALUATION
            
            self.history['losses'].append(avg_loss)
            self.history['kl_divs'].append(avg_kl if self.cfg.algo == "grpo" else 0.0)
            self.history['active_temps'].append(active_temp)
            
            b_size, b_mean_rew, b_max_rew = self.buffer.stats()
            self.history['buffer_mins'].append(-b_max_rew if b_size > 0 else 0.0)
            
            # Convert positive rewards back to negative energies for display
            b_mean_E = -b_mean_rew if b_size > 0 else 0.0
            b_min_E = -b_max_rew if b_size > 0 else 0.0

            print_training_step(
                epoch=epoch, 
                gen_min=gen_min, 
                gen_mean=gen_mean, 
                unique_cnt=unique_cnt, 
                total_seqs=len(tokens), 
                active_temp=active_temp, 
                algo=self.cfg.algo, 
                avg_loss=avg_loss, 
                avg_kl=avg_kl,
                b_size=b_size,
                b_mean_E=b_mean_E,
                b_min_E=b_min_E,
                b_max_size=current_buffer_size
            )
            
            # Evaluation checks periodically
            if epoch % self.cfg.eval_iter == 0:
                self.evaluate(epoch)
        self.history['best_eval_epoch'] = self.best_eval_epoch
        print("\nTraining complete. Saving artifacts...")
        save_training_artifacts(self.history, self.model, self.opt, self.cfg, self.grd_E)
        
if __name__ == "__main__":
    cfg = TrainConfig()
    trainer = RLTrainer(cfg)
    trainer.train()