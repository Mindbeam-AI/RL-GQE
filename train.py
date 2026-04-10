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
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.setup_env()
        self.setup_physics()
        self.setup_models()
        
        self.buffer = EliteReplayBuffer(max_size=cfg.buffer_size, absolute_floor=cfg.buffer_floor)
            
        # Tracking dictionaries
        self.history = {
            'losses': [], 'eval_Es': [], 'eval_epochs': [], 
            'kl_divs': [], 'active_temps': [], 'buffer_mins': []
        }
        self.best_min = float('inf')
        self.best_eval_epoch = 0

    def setup_env(self):
        os.environ['PYTHONHASHSEED'] = str(self.cfg.seed)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def setup_physics(self):
        data = joblib.load(f'./VQE-generated-dataset/data/ground_state/0{self.cfg.num_qubits}qubit/label{self.cfg.ham_label}.jb')
        self.grd_E = data["ground_energy"]
        self.ham = gen_hamiltonian(self.cfg.ham_label, self.cfg.num_qubits)
        self.init_state = [0] * self.cfg.num_qubits
        self.op_pool = np.array(build_operator_pool(self.cfg.num_qubits), dtype=object)

        # Calculate the baseline energy before any gates are applied
        empty_seq = np.array([], dtype=object)
        base_E_array = get_subsequence_energies(empty_seq, self.ham, self.init_state, self.cfg.num_qubits)
        self.base_E = base_E_array[-1] if len(base_E_array) > 0 else 0.0

        # Build Action Mask
        vocab_size = len(self.op_pool) + 1
        self.action_mask = torch.zeros((vocab_size, vocab_size), dtype=torch.bool, device="cuda")

        for i, op1 in enumerate(self.op_pool):
            idx1 = i + 1  # Offset by 1 because 0 is the start token
            for j, op2 in enumerate(self.op_pool):
                idx2 = j + 1
                
                # Rule: Forbid consecutive gates operating on the exact same Pauli axis and wires
                if op1.name == op2.name and op1.wires == op2.wires:
                    self.action_mask[idx1, idx2] = True

    def setup_models(self):
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
        current_temp = temp if temp is not None else self.cfg.temp_explore
        
        self.model.eval()
        tokens, _ = self.model.generate(
            n_sequences=self.cfg.seq_gen * self.cfg.gen_iter, # This forms the Group (G)
            max_new_tokens=self.cfg.seq_len, 
            temperature=current_temp,
            device="cuda",
            action_mask=self.action_mask # apply action mask
        )

        gen_op_seq = self.op_pool[(tokens[:, 1:] - 1).cpu().numpy()]
        energies = torch.tensor(
            get_subsequence_energies(gen_op_seq, self.ham, self.init_state, self.cfg.num_qubits),
            device="cuda", dtype=torch.float32
        )
        
        rewards = -energies[:, -1]

        # --- ALGORITHM ROUTING ---
        if self.cfg.algo == "grpo":
            init_E_tensor = torch.full((energies.size(0), 1), self.base_E, device="cuda", dtype=torch.float32)
            full_E_trajectory = torch.cat([init_E_tensor, energies], dim=1)
            
            # 1. Calculate immediate step rewards
            energy_change = full_E_trajectory[:, :-1] - full_E_trajectory[:, 1:]
            gate_cost = 0.005 
            step_rewards = energy_change - gate_cost
            
            # 2. Calculate the Return-to-Go (Cumulative future rewards)
            # This ensures early gates are judged by the final sequence energy they lead to.
            returns = torch.zeros_like(step_rewards)
            returns[:, -1] = step_rewards[:, -1]
            for t in reversed(range(step_rewards.size(1) - 1)):
                returns[:, t] = step_rewards[:, t] + returns[:, t+1] # Assuming gamma discount = 1.0 for short circuits
            
            # 3. Group Normalization on the Returns
            mean_ret = returns.mean(dim=0, keepdim=True)
            std_ret = returns.std(dim=0, keepdim=True) + 1e-8
            
            final_advs = (returns - mean_ret) / std_ret

            if epoch % self.cfg.ref_sync_iter == 0:
                self.old_model.load_state_dict(self.model.state_dict())
            
            # --- SIL Buffer Push ---
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
            
            # Expand PPO's 1D advantage to match token sequence shape
            final_advs = seq_advs.unsqueeze(-1).expand_as(tokens[:, :-1])
            
        return tokens, final_advs, rewards

    def evaluate(self, epoch):
        self.model.eval()
        tokens, _ = self.model.generate(
            n_sequences=10, 
            max_new_tokens=self.cfg.seq_len, 
            temperature=self.cfg.temp_eval, 
            device="cuda", 
            action_mask=self.action_mask # apply action mask
        )

        eval_op_seq = self.op_pool[(tokens[:, 1:] - 1).cpu().numpy()]
        true_Es = get_subsequence_energies(eval_op_seq, self.ham, self.init_state, self.cfg.num_qubits)[:, -1].reshape(-1, 1)
        
        eval_mean = np.mean(true_Es)
        eval_min = np.min(true_Es)
        unique_cnt = torch.unique(tokens, dim=0).size(0)

        self.history['eval_Es'].append(true_Es)
        self.history['eval_epochs'].append(epoch)

        # Call the external tracking function
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

    def update_policy(self, tokens, advantages, epoch, rewards):
        self.model.train()
        loss_record = 0.0
        kl_record = 0.0 
        
        for _ in range(self.cfg.ppo_epochs):
            self.opt.zero_grad()
            
            if self.cfg.algo == "grpo":
                # 1. The GRPO Explorer
                loss_grpo, kl_mean = compute_grpo_loss(
                    self.model, tokens, advantages, self.cfg.epsilon, self.old_model, self.cfg.beta
                )
                kl_record += kl_mean.item()
                
                # 2. The SIL Anchor
                # Sample from the buffer if it has good circuits
                loss_sil = torch.tensor(0.0, device="cuda")
                if self.buffer is not None and not self.buffer.tokens is None:
                    dynamic_floor = self.buffer.rewards[len(self.buffer.rewards)//2].item() if self.buffer.is_full() else self.cfg.buffer_floor
                    sil_toks, sil_rews = self.buffer.sample(self.cfg.seq_gen)
                    if sil_toks is not None:
                        loss_sil = compute_sil_loss(self.model, sil_toks, sil_rews, floor=dynamic_floor)
                
                # Combine them. SIL weight pulls hard, GRPO refines.
                loss = loss_grpo + (self.cfg.sil_weight * loss_sil)
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                loss_record += loss.item()
                    
        # Return averaged metrics
        avg_loss = loss_record / self.cfg.ppo_epochs
        avg_kl = (kl_record / self.cfg.ppo_epochs) if self.cfg.algo == "grpo" else 0.0
        return avg_loss, avg_kl

    def fill_initial_buffer(self):
        print("\n--- Filling Initial Buffer ---")
        fill_step = 1
        while not self.buffer.is_full():
            self.generate_rollouts(epoch=0)
            b_size, b_mean, b_max = self.buffer.stats()
            print(f"Fill Step {fill_step} | Buffer: {b_size}/{self.cfg.buffer_size} | Mean E: {-b_mean:.4f} | Best E: {-b_max:.4f}")
            fill_step += 1
        print("--- Buffer Full. Starting Training ---\n")

    def train(self):
        if self.cfg.algo == "ppo_sil":
            self.fill_initial_buffer()
        
        step_temp = 0.0
        spike_temp = 0.0
        stagnation_counter = 0
        best_window_min = float('inf')
        
        for epoch in range(self.cfg.n_epochs + 1):
            active_temp = min(self.cfg.temp_min + step_temp + spike_temp, self.cfg.temp_max)
            
            # --- 1. GENERATION ---
            if epoch % self.cfg.gen_iter == 0:
                tokens, advantages, rewards = self.generate_rollouts(epoch=epoch, temp=active_temp)
                
                gen_energies = -rewards
                gen_mean = gen_energies.mean().item()
                gen_min = gen_energies.min().item()
                gen_std = gen_energies.std().item() # NEW: Calculate standard deviation
                unique_cnt = torch.unique(tokens, dim=0).size(0)
                
                # --- NEW TEMPERATURE LOGIC (Energy Stagnation) ---
                spike_temp = 0.0 
                
                # Check if we found a meaningfully better minimum
                if gen_min < best_window_min - 0.02: 
                    best_window_min = gen_min
                    stagnation_counter = 0
                    
                    # Cool down: we found a valley, let's exploit it
                    if step_temp > 0.0:
                        step_temp = max(step_temp - self.cfg.temp_step * 2, 0.0)
                else:
                    stagnation_counter += 1

                # If stuck in the same energy band for 15 epochs
                if stagnation_counter >= 15:
                    spike_temp = self.cfg.spike_magnitude
                    # Raise the baseline temperature slightly to encourage wider search
                    step_temp = min(step_temp + self.cfg.temp_step, self.cfg.temp_max - self.cfg.temp_min)
                    
                    stagnation_counter = 0 # Reset the counter
                    best_window_min = gen_min # Reset the baseline so it doesn't get permanently stuck
                    

            # --- 2. OPTIMIZATION ---
            avg_loss, avg_kl = self.update_policy(tokens, advantages, epoch, rewards)
            
            # Logging
            self.history['losses'].append(avg_loss)
            self.history['kl_divs'].append(avg_kl if self.cfg.algo == "grpo" else 0.0)
            self.history['active_temps'].append(active_temp)
            
            # --- 3. PRINT TRAINING STEP ---
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
                b_max_size=self.cfg.buffer_size
            )
            
            # --- 4. EVALUATION (Runs after training print) ---
            if epoch % self.cfg.eval_iter == 0:
                self.evaluate(epoch)
        self.history['best_eval_epoch'] = self.best_eval_epoch
        print("\nTraining complete. Saving artifacts...")
        save_training_artifacts(self.history, self.model, self.opt, self.cfg, self.grd_E)
        
if __name__ == "__main__":
    cfg = TrainConfig()
    trainer = RLTrainer(cfg)
    trainer.train()