import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import pennylane as qml
import copy
import json
import matplotlib.pyplot as plt
import pandas as pd
import holoviews as hv
import hvplot.pandas

from GPTQE import GPTQE
from model import GPTConfig

def calculate_loss_GRPO(model, tokens, advantages, epsilon, old_model=None):
    # --- Step 1: Current policy distribution ---
    logits = model(tokens[:, :-1])
    log_probs_all = F.log_softmax(logits, dim=-1)
    probs_all = F.softmax(logits, dim=-1)
    
    # Extract log-probs of the actions actually taken
    curr_log_probs = log_probs_all.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1) 

    # --- Step 2: Old policy ---
    if old_model is not None:
        with torch.no_grad():
            old_logits = old_model(tokens[:, :-1])
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_log_probs = old_log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    else:
        old_log_probs = curr_log_probs.detach()

    # --- Step 3: Ratios & Clipping ---
    log_ratio = curr_log_probs - old_log_probs
    log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
    ratios = torch.exp(log_ratio)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

    # --- Step 4: EXACT ENTROPY BONUS ---
    # This prevents the model from collapsing to a single sequence
    # by rewarding it for keeping its probability distribution spread out.
    entropy = -torch.sum(probs_all * log_probs_all, dim=-1).mean()
    entropy_coef = 0.01 # Adjust this if the model is too random or too rigid

    # Maximize surrogate, maximize entropy -> Minimize negative surrogate, minimize negative entropy
    loss = -torch.mean(torch.min(surr1, surr2)) - (entropy_coef * entropy)
    
    return loss
    
def build_operator_pool(n_qubits, t_values=None):

    if t_values is None:
        # t_values = [np.pi, np.pi/2, np.pi/3, np.pi/4, np.pi/8]
        t_values = [np.pi, np.pi/2, np.pi/4, np.pi/8, np.pi/16, np.pi/32]
        t_values += [-t for t in t_values]  # add negatives

    pool = []

    # Two-qubit interactions (Z_i Z_{i+1})
    for i in range(n_qubits - 1):
        for t in t_values:
            pool.append(qml.PauliRot(t, 'ZZ', wires=[i, i + 1]))
            pool.append(qml.PauliRot(t, 'XX', wires=[i, i+1]))
            pool.append(qml.PauliRot(t, 'YY', wires=[i, i+1]))

    # Single-qubit terms (X_i)
    for i in range(n_qubits):
        for t in t_values:
            pool.append(qml.PauliRot(t, 'X', wires=[i]))
            pool.append(qml.PauliRot(t, 'Y', wires=i))
            pool.append(qml.PauliRot(t, 'Z', wires=i))

    # for i in range(n_qubits - 2):
    #     for t in t_values:
    #         pool.append(qml.PauliRot(2 * t, 'ZZ', wires=[i, i + 2]))
    #         pool.append(qml.PauliRot(2 * t, 'XX', wires=[i, i+2]))
    #         pool.append(qml.PauliRot(2 * t, 'YY', wires=[i, i+2]))

    return pool

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def energy_circuit(seq, ham, init_state, num_qubits):
    qml.BasisState(init_state, wires=range(num_qubits))

    for op in seq:
        qml.Snapshot(measurement=qml.expval(ham))
        qml.apply(op)
    return qml.expval(ham)

energy_circuit = qml.snapshots(energy_circuit)

def get_subsequence_energies(seq, hamiltonian, init_state, num_qubits):
    energies = []
    for pool in seq:
        es = energy_circuit(pool, hamiltonian, init_state, num_qubits)
        energies.append(
            [es[k].item() for k in list(range(1, len(pool))) + ["execution_results"]]
        )
    return np.array(energies, dtype=np.float32)

def gen_hamiltonian(ham_label, num_qubits):
    # User implementation expected here
    obs = [qml.PauliZ(0) @ qml.PauliZ(1)]
    coeffs = [1.0]
    return qml.Hamiltonian(coeffs, obs)

def main():
    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- HYPERPARAMETERS ---
    ham_label = 1
    num_qubits = 4
    seq_gen = 80 
    seq_len = 12
    gen_iter = 1
    temperature = 2.0
    n_epochs = 700
    n_batches = 10
    epsilon = 0.2
    load_model = False

    # Target directory for saves
    dir = "./experiments/heisenberg_extraangles_withbeta_grpo"
    os.makedirs(f"{dir}/histo", exist_ok=True)

    # --- SETUP ---
    data = joblib.load(f'./VQE-generated-dataset/data/ground_state/0{num_qubits}qubit/label{ham_label}.jb')
    grd_E = data["ground_energy"]
    ham = gen_hamiltonian(ham_label, num_qubits)
    init_state = [0] * num_qubits
    op_pool = np.array(build_operator_pool(num_qubits), dtype=object)
    op_pool_size = len(op_pool)

    

    # --- MODEL INSTANTIATION ---
    if load_model:
        load_dir = "./saved_models"
        model = GPTQE(GPTConfig(
            vocab_size=op_pool_size + 1,
            block_size=seq_len,
            dropout=0.2,
            bias=False,
            n_layer=4,
            n_embd=384,
            n_head=8,
        )).to("cuda")
        
        load = torch.load(f"{load_dir}/final.pt", map_location="cpu")
        model.load_state_dict(load["model_state_dict"])
        opt = model.configure_optimizers(
            weight_decay=0.01, learning_rate=4e-4, betas=(0.9, 0.95), device_type="auto"
        )
        opt.load_state_dict(load["optimizer_state_dict"])
    else:
        model = GPTQE(GPTConfig(
            vocab_size=op_pool_size + 1,
            block_size=seq_len,
            dropout=0.2,
            bias=False,
            n_layer=4,
            n_head=8,
            n_embd=384
        )).to("cuda")

        opt = model.configure_optimizers(
            weight_decay=0.01, learning_rate=4e-4, betas=(0.9, 0.95), device_type="auto"
        )

    # --- GRPO REFERENCE MODEL SETUP ---
    old_model = copy.deepcopy(model)
    for param in old_model.parameters():
        param.requires_grad = False
    old_model.eval()

    # --- TRACKING ---
    losses = []
    true_Es_g = []
    pred_Es_t = []
    true_Es_t = []
    eval_iter = 50
    current_mae = float('inf')
    best_min = float('inf')
    min_epoch = 0

    # --- TRAINING LOOP ---
    for epoch in range(0, n_epochs + 1):
        
        # 1. GENERATION PHASE
        if epoch % gen_iter == 0:
            model.eval()
            with torch.no_grad():
                tokens, _ = model.generate(
                    n_sequences=int(seq_gen * gen_iter),
                    max_new_tokens=seq_len,
                    temperature=temperature,
                    device="cuda"
                )

            gen_inds = (tokens[:, 1:] - 1).cpu().numpy()
            gen_op_seq = op_pool[gen_inds]

            energies = torch.from_numpy(
                get_subsequence_energies(gen_op_seq, ham, init_state, num_qubits)
            ).to("cuda")
            
            true_Es_g.append(energies[:, -1].cpu().numpy().reshape(-1, 1))
            
            # Calculate advantages across all 80 sequences to ensure stable statistics
            rewards = -energies
            mean_r = rewards.mean(dim=0, keepdim=True)
            std_r = rewards.std(dim=0, unbiased=False, keepdim=True) + 1e-5
            advantages = (rewards - mean_r) / std_r
            
            # Synchronize old model
            old_model.load_state_dict(model.state_dict())

        # 2. EVALUATION & LOGGING PHASE
        if epoch % eval_iter == 0:
            model.eval()
            # Evaluate current policy
            true_Es = energies[:, -1].cpu().numpy().reshape(-1, 1)
            
            # In RL, the policy's achieved energy IS the predicted outcome.
            pred_Es = true_Es 
            
            mae = np.mean(np.abs(pred_Es - grd_E)) # Comparing to ground energy instead of regression target
            ave_E = np.mean(true_Es)
            min_E = np.min(true_Es)
            
            if min_E < best_min:
                best_min = min_E
                min_epoch = epoch
            current_mae = mae

            pred_Es_t.append(pred_Es)
            true_Es_t.append(true_Es)

            curr_loss = losses[-1] if len(losses) > 0 else 0.0
            print(f"Iteration: {epoch}, Loss: {curr_loss:.4f}, MAE: {mae:.4f}, Ave E: {ave_E:.4f}, Min E: {min_E:.4f}")

            # Plotting Histogram
            plt.figure(figsize=(10, 5))
            plt.hist(pred_Es, bins=30, alpha=0.6, label='Current Policy Energy')
            plt.axvline(min_E, color='red', linestyle='--', label='Min Policy E')
            plt.axvline(ave_E, color='black', linestyle='--', label='Average Policy E')
            plt.axvline(grd_E, color='green', linestyle='-', linewidth=2, label='True Ground State E')
            plt.legend()
            plt.title(f"Energy Distribution @ Epoch {epoch}")
            plt.xlabel("Energy")
            plt.ylabel("Count")
            plt.savefig(f"{dir}/histo/{epoch}.png")
            plt.close()

        # 3. OPTIMIZATION PHASE
        model.train()
        train_inds = np.arange(len(tokens))
        np.random.shuffle(train_inds)
        
        # Split both tokens and advantages
        token_batches = torch.tensor_split(tokens[train_inds], n_batches)
        advantage_batches = torch.tensor_split(advantages[train_inds], n_batches)
                
        loss_record = 0

        for token_batch, advantage_batch in zip(token_batches, advantage_batches):
            opt.zero_grad()
            
            # Since all sequences share the same initial quantum state, the entire mini-batch 
            # acts as a single group. We dynamically pass the mini-batch length.
            current_batch_size = len(token_batch)
            
            loss = calculate_loss_GRPO(
                model=model,
                tokens=token_batch, 
                advantages=advantage_batch, # Pass the pre-calculated batch
                epsilon=epsilon, 
                old_model=old_model
            )
            
            loss.backward()
            
            # Physically prevents logit explosion even if advantages spike
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            opt.step()
            loss_record += loss.item()

        avg_loss = loss_record / n_batches
        losses.append(avg_loss)
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Mean Energy: {energies[:, -1].mean().item():.4f} | Min Energy: {energies[:, -1].min().item():.4f} ")

    
    # ==========================================
    # POST-TRAINING SAVING & PLOTTING
    # ==========================================
    df_loss = pd.DataFrame(losses, columns=["loss"])
    df_loss.to_csv(f"{dir}/losses.csv", index=False)

    hvplot.extension('matplotlib')

    loss_fig = df_loss.hvplot(
        title="Training loss progress", ylabel="loss", xlabel="Training epochs"
    ).opts(fig_size=600, fontscale=2, aspect=1.2)
    hv.save(loss_fig, f"{dir}/loss_fig.png")

    pred_Es_t = np.concatenate(pred_Es_t, axis=1)
    true_Es_t = np.concatenate(true_Es_t, axis=1)

    df_pred = pd.DataFrame(pred_Es_t, columns=list(range(0, n_epochs+1, eval_iter)))
    df_true = pd.DataFrame(true_Es_t, columns=list(range(0, n_epochs+1, eval_iter)))

    df_pred.to_csv(f"{dir}/pred_Es_t.csv", index=False)
    df_true.to_csv(f"{dir}/true_Es_t.csv", index=False)

    df_trues_stats = pd.concat([df_true.mean(axis=0), df_true.min(axis=0), df_true.max(axis=0)], axis=1).reset_index()
    df_trues_stats.columns = ["Training Iterations", "Ave True E", "Min True E", "Max True E"]

    df_preds_stats = pd.concat([df_pred.mean(axis=0), df_pred.min(axis=0), df_pred.max(axis=0)], axis=1).reset_index()
    df_preds_stats.columns = ["Training Iterations", "Ave Pred E", "Min Pred E", "Max Pred E"]

    fig = (
        df_trues_stats.hvplot.scatter(x="Training Iterations", y="Ave True E", label="Mean True Energies") *
        df_trues_stats.hvplot.line(x="Training Iterations", y="Ave True E", alpha=0.5, linewidth=1) *
        df_trues_stats.hvplot.area(x="Training Iterations", y="Min True E", y2="Max True E", alpha=0.1)
    ) * (
        df_preds_stats.hvplot.scatter(x="Training Iterations", y="Ave Pred E", label="Mean Predicted Energies") *
        df_preds_stats.hvplot.line(x="Training Iterations", y="Ave Pred E", alpha=0.5, linewidth=1) *
        df_preds_stats.hvplot.area(x="Training Iterations", y="Min Pred E", y2="Max Pred E", alpha=0.1)
    )
    fig = fig * hv.Curve([[0, grd_E], [n_epochs+1, grd_E]], label="Ground State Energy").opts(color="k", alpha=0.4, linestyle="dashed")
    fig = fig.opts(ylabel="Sequence Energies", title="GQE Evaluations", fig_size=600, fontscale=2)
    hv.save(fig, f"{dir}/eval_fig.png")

    true_Es_g = np.concatenate(true_Es_g, axis=1)
    df_gen = pd.DataFrame(true_Es_g)
    df_gen.columns = df_gen.columns.astype(int)
    
    df_gen_stats = pd.concat([df_gen.mean(axis=0), df_gen.min(axis=0), df_gen.max(axis=0)], axis=1).reset_index()
    df_gen_stats.columns = ["Training Iterations", "Ave True E", "Min True E", "Max True E"]
    
    gen_fig = (
        df_gen_stats.hvplot.scatter(x="Training Iterations", y="Ave True E", label="Mean True Energies") *
        df_gen_stats.hvplot.line(x="Training Iterations", y="Ave True E", alpha=0.5, linewidth=1) *
        df_gen_stats.hvplot.area(x="Training Iterations", y="Min True E", y2="Max True E", alpha=0.1)
    )
    gen_fig = gen_fig.opts(ylabel="Sequence Energies", title="Generated Training Data", fig_size=600, fontscale=2)
    hv.save(gen_fig, f"{dir}/gen_fig.png")

    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
    }
    torch.save(save_dict, f"{dir}/final.pt")

    with open(f"{dir}/config.json", "w") as f:
        json.dump(model.config.__dict__, f, indent=4)

    metadata = {
        "num_qubits": num_qubits,
        "ham_label": ham_label,
        "epochs": n_epochs,
        "seq len": seq_len,
        "seq gen": seq_gen,
        "gen iter": gen_iter,
        "n batches": n_batches,
        "weight beta": beta,
        "lowest energy": best_min,
        "final_loss": float(loss.item() if torch.is_tensor(loss) else loss),
        "min_mae": float(current_mae),
        "min_mae_epoch": min_epoch,
        "temperature": temperature,
    }

    with open(f"{dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    main()