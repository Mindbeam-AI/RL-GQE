import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holoviews as hv
import hvplot.pandas
from dataclasses import asdict

def plot_epoch_histogram(energies, epoch, grd_E, save_dir):
    plt.figure(figsize=(10, 5))
    
    # In RL, we only have the measured sequence energy, not a predicted scalar
    plt.hist(energies, bins=30, alpha=0.7, color='blue', edgecolor='black', label='Generator Energy')
    
    plt.axvline(np.min(energies), color='red', linestyle='--', label=f'Min E: {np.min(energies):.4f}')
    plt.axvline(np.mean(energies), color='black', linestyle='--', label=f'Mean E: {np.mean(energies):.4f}')
    
    if grd_E is not None:
        plt.axvline(grd_E, color='green', linestyle='-', linewidth=2, label=f'Ground Truth: {grd_E:.4f}')
        
    plt.legend()
    plt.title(f"Generator Energy Distribution @ Epoch {epoch}")
    plt.xlabel("Energy")
    plt.ylabel("Count")
    
    os.makedirs(f"{save_dir}/histo", exist_ok=True)
    plt.savefig(f"{save_dir}/histo/epoch_{epoch}.png")
    plt.close()

def save_training_artifacts(history, model, optimizer, cfg, grd_E=None):
    dir = cfg.save_dir
    os.makedirs(dir, exist_ok=True)
    hvplot.extension('matplotlib')

    # 1. Training Dynamics (Loss, KL, Temp, Buffer)
    if history.get('losses'):
        # Combine all per-epoch metrics into one DataFrame
        df_dynamics = pd.DataFrame({
            "Epoch": range(1, len(history['losses']) + 1),
            "Loss": history['losses'],
            "KL_Div": history.get('kl_divs', []),
            "Temperature": history.get('active_temps', []),
            "Buffer_Min": history.get('buffer_mins', [])
        })
        df_dynamics.to_csv(f"{dir}/training_dynamics.csv", index=False)
        
        # Plot Loss
        loss_fig = df_dynamics.hvplot(
            x="Epoch", y="Loss", title="Training Loss", ylabel="Loss"
        ).opts(fig_size=600, fontscale=2, aspect=1.2)
        hv.save(loss_fig, f"{dir}/loss_fig.png")

        # Plot Temperature & Buffer Progress
        temp_curve = df_dynamics.hvplot.line(
            x="Epoch", y="Temperature", ylabel="Temperature", title="Active Temperature", color="orange"
        )
        buffer_curve = df_dynamics.hvplot.line(
            x="Epoch", y="Buffer_Min", ylabel="Min Energy", title="Buffer Progression", color="purple"
        )
        
        # Combine into a vertical layout (1 column)
        diag_fig = (temp_curve + buffer_curve).cols(1).opts(shared_axes=False, fig_size=300, fontscale=1.5)
        hv.save(diag_fig, f"{dir}/diagnostics_fig.png")

    # 2. Evaluation Progress (Deterministic Policy)
    if history.get('eval_Es'):
        eval_Es = np.concatenate(history['eval_Es'], axis=1)
        eval_epochs = history.get('eval_epochs', list(range(cfg.eval_iter, cfg.n_epochs + 1, cfg.eval_iter)))
        
        df_eval = pd.DataFrame(eval_Es, columns=eval_epochs)
        df_eval.to_csv(f"{dir}/eval_Es.csv", index=False)
        
        df_stats = pd.concat([df_eval.mean(axis=0), df_eval.min(axis=0), df_eval.max(axis=0)], axis=1).reset_index()
        df_stats.columns = ["Epoch", "Mean E", "Min E", "Max E"]
        
        fig = (
            df_stats.hvplot.scatter(x="Epoch", y="Mean E", label="Mean Policy Energy", color="blue") *
            df_stats.hvplot.line(x="Epoch", y="Mean E", alpha=0.5, linewidth=1, color="blue") *
            df_stats.hvplot.area(x="Epoch", y="Min E", y2="Max E", alpha=0.2, color="blue")
        )
        if grd_E is not None:
            fig = fig * hv.Curve([[0, grd_E], [cfg.n_epochs, grd_E]], label="Ground State").opts(color="red", linestyle="dashed")
            
        fig = fig.opts(ylabel="Energy", title="RL Policy Evaluation Progress", fig_size=600, fontscale=2)
        hv.save(fig, f"{dir}/eval_fig.png")

    # 3. Model Checkpoint
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(save_dict, f"{dir}/final.pt")

    # 4. Configurations & Metadata
    with open(f"{dir}/config.json", "w") as f:
        json.dump(model.config.__dict__, f, indent=4)

    metadata = asdict(cfg)
    
    # Append final RL stats if available
    metadata.update({
        "ground_energy": float(grd_E) if grd_E is not None else None,
        "final_loss": float(history['losses'][-1]) if history.get('losses') else None,
        "best_eval_min": float(np.min(eval_Es)) if history.get('eval_Es') else None,
        "best_eval_epoch": int(history.get('best_eval_epoch', 0)) 
    })

    with open(f"{dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

def print_training_step(epoch, gen_min, gen_mean, unique_cnt, total_seqs, active_temp, algo, avg_loss, avg_kl=None, b_size=0, b_mean_E=0.0, b_min_E=0.0, b_max_size=24):
    print(f"Epoch {epoch} | Gen Min: {gen_min:.4f} | Gen Mean: {gen_mean:.4f} | Unique: {unique_cnt}/{total_seqs} | Temp Used: {active_temp:.2f}")
    
    if b_size > 0:
        print(f"  [Buffer] Size: {b_size}/{b_max_size} | Mean E: {b_mean_E:.4f} | Min E: {b_min_E:.4f}")
    else:
        print(f"  [Buffer] Empty (Waiting for E < -{4.00})")
        
    if algo == "grpo" and avg_kl is not None:
        print(f"  [Optimization] Loss: {avg_loss:.4f} | KL Div: {avg_kl:.4f}")
    else:
        print(f"  [Optimization] Loss: {avg_loss:.4f}")

def print_eval_step(epoch, eval_min, eval_mean, unique_cnt, total_seqs, temp_eval, eval_seq):
    print(f"[Verify] Eval Min: {eval_min:.4f} | Eval Mean: {eval_mean:.4f} | Unique: {unique_cnt}/{total_seqs} | Temp Used: {temp_eval:.2f}")
    print(f"         Eval Seq: {eval_seq}\n")