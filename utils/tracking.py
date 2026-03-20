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

    # 1. Losses
    if history.get('losses'):
        df_loss = pd.DataFrame(history['losses'], columns=["loss"])
        df_loss.to_csv(f"{dir}/losses.csv", index=False)
        loss_fig = df_loss.hvplot(
            title="Training Loss Progress", ylabel="Loss", xlabel="Optimization Steps"
        ).opts(fig_size=600, fontscale=2, aspect=1.2)
        hv.save(loss_fig, f"{dir}/loss_fig.png")

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
        "best_eval_min": float(np.min(eval_Es)) if history.get('eval_Es') else None
    })

    with open(f"{dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)