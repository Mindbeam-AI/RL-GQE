import os
import random
import numpy as np
import torch
import copy
import joblib

# Modular Imports
from config import TrainConfig
from physics import gen_hamiltonian, build_operator_pool, get_subsequence_energies
from rl import EliteReplayBuffer, compute_ppo_loss, compute_sil_loss
from models.model import GPTConfig
from models.GPTQE import GPTQE
from utils.tracking import plot_epoch_histogram, save_training_artifacts

class RLTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.setup_env()
        self.setup_physics()
        self.setup_models()
        self.buffer = EliteReplayBuffer(max_size=cfg.buffer_size, absolute_floor=cfg.buffer_floor)
        
        # Tracking dictionaries
        self.history = {'losses': [], 'eval_Es': [], 'eval_epochs': []}
        self.best_min = float('inf')

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

    def generate_rollouts(self, temp=None):
        # 1. Use the passed temp, or fallback to default if none is provided
        current_temp = temp if temp is not None else self.cfg.temp_explore
        
        self.model.eval()
        tokens, _ = self.model.generate(
            n_sequences=self.cfg.seq_gen * self.cfg.gen_iter,
            max_new_tokens=self.cfg.seq_len, 
            temperature=current_temp,
            device="cuda"
        )

        gen_op_seq = self.op_pool[(tokens[:, 1:] - 1).cpu().numpy()]
        energies = torch.tensor(
            get_subsequence_energies(gen_op_seq, self.ham, self.init_state, self.cfg.num_qubits),
            device="cuda", dtype=torch.float32
        )
        
        rewards = -energies[:, -1]
        
        # 1. Define the Global Baseline (The worst circuit in the buffer)
        if self.buffer.is_full():
            global_baseline = self.buffer.rewards[-1].item()
        else:
            global_baseline = self.cfg.buffer_floor

        # 2. Strict Global Filtering
        seq_advs = torch.zeros_like(rewards)
        mask = rewards > global_baseline
        
        # 3. Scale advantages proportionally to prevent "Frankenstein" averaging
        if mask.sum() > 0:
            seq_advs[mask] = torch.clamp(rewards[mask] - global_baseline, min=0.1, max=3.0)
        
        self.buffer.add(tokens, rewards, seq_advs)
        self.old_model.load_state_dict(self.model.state_dict())
        
        return tokens, seq_advs.unsqueeze(-1).expand_as(tokens[:, :-1]), rewards

    def evaluate(self, epoch):
        self.model.eval()
        tokens, _ = self.model.generate(
            n_sequences=10, max_new_tokens=self.cfg.seq_len, temperature=self.cfg.temp_eval, device="cuda"
        )

        eval_op_seq = self.op_pool[(tokens[:, 1:] - 1).cpu().numpy()]
        true_Es = get_subsequence_energies(eval_op_seq, self.ham, self.init_state, self.cfg.num_qubits)[:, -1].reshape(-1, 1)
        
        eval_mean = np.mean(true_Es)
        eval_std = np.std(true_Es)
        eval_min = np.min(true_Es)
        eval_max = np.max(true_Es)
        unique_cnt = torch.unique(tokens, dim=0).size(0)

        self.best_min = min(self.best_min, eval_min)
        self.history['eval_Es'].append(true_Es)
        self.history['eval_epochs'].append(epoch)

        print(f"\n--- EVALUATION @ EPOCH {epoch} ---")
        print(f"EVAL E -> Mean: {eval_mean:.4f} | Std: {eval_std:.4f} | Min: {eval_min:.4f} | Max: {eval_max:.4f} | Unique: {unique_cnt}/{len(tokens)}")
        print(f"  [Verify] Buffer Seq 0: {self.buffer.tokens[0, 1:].cpu().numpy()}")
        print(f"  [Verify] Eval Seq 0:   {tokens[0, 1:].cpu().numpy()}")
        print("----------------------------------\n")

        if epoch % self.cfg.plot_iter == 0:
            plot_epoch_histogram(true_Es.flatten(), epoch, self.grd_E, self.cfg.save_dir)

    def update_policy(self, tokens, advantages, epoch, rewards):
        self.model.train()
        
        # Calculate a dynamic floor: Only learn from the top 50% of the buffer
        if self.buffer.is_full():
            # Use the median reward of the buffer as the new floor
            dynamic_floor = self.buffer.rewards[len(self.buffer.rewards)//2].item()
        else:
            dynamic_floor = self.cfg.buffer_floor

        # SIL Weight: Stay high to ensure I don't forget the best circuits
        sil_weight = 2.0 
        policy_weight = 1.0 if advantages.max().item() > 0.0 else 0.0
        
        loss_record = 0.0
        for _ in range(self.cfg.ppo_epochs):
            self.opt.zero_grad()
            
            # 1. On-Policy PPO - The Scout
            loss_ppo = compute_ppo_loss(self.model, tokens, advantages, self.cfg.epsilon, self.old_model, self.cfg.entropy_coef) if policy_weight > 0 else torch.tensor(0.0, device="cuda")
            
            # 2. Self-Imitation - Positive Only Loss on Buffer
            sil_toks, sil_rews = self.buffer.sample(self.cfg.seq_gen)
            loss_sil = compute_sil_loss(self.model, sil_toks, sil_rews, floor=dynamic_floor) if sil_toks is not None else torch.tensor(0.0, device="cuda")

            loss = (policy_weight * loss_ppo) + (sil_weight * loss_sil)
            
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                loss_record += loss.item()
                    
        return loss_record

    def fill_initial_buffer(self):
        print("\n--- Filling Initial Buffer ---")
        fill_step = 1
        while not self.buffer.is_full():
            self.generate_rollouts()
            b_size, b_mean, b_max = self.buffer.stats()
            print(f"Fill Step {fill_step} | Buffer: {b_size}/{self.cfg.buffer_size} | Mean E: {-b_mean:.4f} | Best E: {-b_max:.4f}")
            fill_step += 1
        print("--- Buffer Full. Starting Training ---\n")

    def train(self):
        self.fill_initial_buffer()
        
        step_temp = 0.0
        spike_temp = 0.0
        
        for epoch in range(self.cfg.n_epochs + 1):
            
            # 1. Calculate Active Temp and clamp it to the ceiling
            active_temp = min(self.cfg.temp_min + step_temp + spike_temp, self.cfg.temp_max)
            
            # 2. Explore
            if epoch % self.cfg.gen_iter == 0:
                tokens, advantages, rewards = self.generate_rollouts(temp=active_temp)
                b_size, b_mean, b_max = self.buffer.stats()
                
                gen_energies = -rewards
                gen_mean = gen_energies.mean().item()
                unique_cnt = torch.unique(tokens, dim=0).size(0)
                
                # Reset temporary spike after use
                spike_temp = 0.0 
                
                # --- Bidirectional Temperature Logic ---
                diversity_ratio = unique_cnt / len(tokens)
                
                # Condition A: Policy Collapsing -> Heat Up
                if diversity_ratio < 0.20:
                    spike_temp = self.cfg.spike_magnitude
                    # Ratchet up, but do not exceed the distance to temp_max
                    step_temp = min(step_temp + self.cfg.temp_step, self.cfg.temp_max - self.cfg.temp_min)
                    print(f"  [Heating] Unique: {unique_cnt}. Ratcheting base. Next Temp will spike.")
                
                # Condition B: Thermal Meltdown -> Cool Down
                elif diversity_ratio >= 0.90 and step_temp > 0.0:
                    # Decay down, but never drop below 0.0 (which keeps active_temp at temp_min)
                    step_temp = max(step_temp - self.cfg.temp_step * 4, 0.0)
                    print(f"  [Cooling] Unique: {unique_cnt}. Diversity restored. Decaying base temp.")
                
                print(f"Epoch {epoch} | Gen Mean: {gen_mean:.4f} | Unique: {unique_cnt}/{len(tokens)} | Temp Used: {active_temp:.2f}")
                print(f"  [Buffer] Size: {b_size}/{self.cfg.buffer_size} | Mean E: {-b_mean:.4f} | Best E: {-b_max:.4f}")

            # 3. Evaluate
            if epoch % self.cfg.eval_iter == 0:
                self.evaluate(epoch)

            # 4. Optimize
            avg_loss = self.update_policy(tokens, advantages, epoch, rewards)
            self.history['losses'].append(avg_loss)
            
            if epoch % self.cfg.eval_iter != 0:
                print(f"  [Optimization] Loss: {avg_loss:.4f}\n")

        print("\nTraining complete. Saving artifacts...")
        save_training_artifacts(self.history, self.model, self.opt, self.cfg, self.grd_E)
        
if __name__ == "__main__":
    cfg = TrainConfig()
    trainer = RLTrainer(cfg)
    trainer.train()