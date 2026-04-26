import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
RUN_DIRS = {
    "Proposed Model (GRPO + Linear SIL)": "best_1",
    "PPO + SIL": "ppo_sil_2",
    "Ablated SIL": "pure_GRPO_3",
    "Exponential SIL": "exponential_SIL_4",
    "Ablated Action Mask": "No_Action_Mask_5",
    "Ablated Discount Factor": "No_Discount_6",
    "Static Buffer": "Static_Buffer_7",
    "Temp Spiking Decay": "temp_spiking_decay_8"
}

BASE_PATH = "../experiments/ablation_studies"
OUTPUT_DIR = "./report_figures"
GROUND_TRUTH_E = -7.8284 # Theoretical ground state

# STYLING
plt.rcParams.update({
    'font.size': 12, 'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white',
    'legend.frameon': True, 'legend.edgecolor': 'black'
})

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# DATA LOADING HELPERS
# ==========================================
def load_eval_data(run_name):
    path = os.path.join(BASE_PATH, RUN_DIRS[run_name], "eval_Es.csv")
    if not os.path.exists(path): return None, None, None
    df = pd.read_csv(path)
    epochs = df.columns.astype(int).values
    mins = df.min(axis=0).values
    means = df.mean(axis=0).values
    return epochs, mins, means

def load_dynamics_data(run_name):
    path = os.path.join(BASE_PATH, RUN_DIRS[run_name], "training_dynamics.csv")
    if not os.path.exists(path): return None
    return pd.read_csv(path)

# ==========================================
# FIGURE 1: BASELINE COMPARISON
# ==========================================
def plot_baseline_comparison():
    fig, ax = plt.subplots(figsize=(9, 6))
    runs = ["Proposed Model (GRPO + Linear SIL)", "PPO + SIL", "Ablated SIL"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for run, col in zip(runs, colors):
        epochs, mins, _ = load_eval_data(run)
        if epochs is not None:
            best_val = np.min(mins)
            ax.plot(epochs, mins, label=f"{run} (Min: {best_val:.4f})", color=col, lw=2.5)

    ax.axhline(y=GROUND_TRUTH_E, color='r', linestyle='--', label=f'Ground Truth ({GROUND_TRUTH_E})', zorder=1)
    ax.set_title("Baseline Algorithm Comparison")
    ax.set_ylabel("Minimum Evaluation Energy")
    ax.set_xlabel("Training Epochs")
    ax.legend(loc="upper right", fontsize=10)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_baseline_comparison.png"), dpi=300)
    plt.close(fig)

# ==========================================
# FIGURE 2: ARCHITECTURAL ABLATIONS (LOG PERCENTAGE ERROR)
# ==========================================
def plot_architectural_ablations():
    fig, ax = plt.subplots(figsize=(9, 6))
    
    configs = [
        {"name": "Proposed Model (GRPO + Linear SIL)", "color": "#1f77b4", "style": "-", "width": 3.0, "z": 10},
        {"name": "Ablated Action Mask", "color": "#d62728", "style": "--", "width": 1.5, "z": 5},
        {"name": "Ablated Discount Factor", "color": "#9467bd", "style": ":", "width": 2.0, "z": 4},
        {"name": "Static Buffer", "color": "#8c564b", "style": "-.", "width": 1.5, "z": 3}
    ]
    
    for cfg in configs:
        epochs, mins, _ = load_eval_data(cfg["name"])
        if epochs is not None:
            # Calculate Percentage Error
            pct_error = (np.abs(mins - GROUND_TRUTH_E) / np.abs(GROUND_TRUTH_E)) * 100
            
            final_pct_error = pct_error[-1]
            ax.plot(epochs, pct_error, label=f"{cfg['name']} (Err: {final_pct_error:.2f}%)", 
                    color=cfg["color"], linestyle=cfg["style"], lw=cfg["width"], zorder=cfg["z"])

    ax.set_yscale('log')
    
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    ax.set_title("Architectural Ablations Convergence Accuracy")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Relative Error (%)")
    
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    ax.legend(loc="upper right", fontsize=9, frameon=True, edgecolor='black')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_architectural_ablations_log_pct.png"), dpi=300)
    plt.close(fig)

# ==========================================
# FIGURE 3: REWARD SHAPING & STABILITY (KL DIV)
# ==========================================
def plot_stability_kldiv():
    fig, ax = plt.subplots(figsize=(9, 6))
    runs = ["Proposed Model (GRPO + Linear SIL)", "Exponential SIL"]
    colors = ['#1f77b4', '#e377c2']
    
    for run, col in zip(runs, colors):
        df = load_dynamics_data(run)
        if df is not None:
            smoothed_kl = df['KL_Div'].rolling(window=5, min_periods=1).mean()
            ax.plot(df['Epoch'], smoothed_kl, label=run, color=col, lw=2)

    ax.set_title("Gradient Stability (KL Divergence)")
    ax.set_ylabel("KL Divergence (Smoothed)")
    ax.set_xlabel("Training Epochs")
    ax.set_yscale('log')
    ax.legend(loc="upper left")
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_kl_stability.png"), dpi=300)
    plt.close(fig)

# ==========================================
# FIGURE 4: TEMPERATURE DYNAMICS (MULTIPLOT)
# ==========================================
def plot_temperature_dynamics():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
    
    runs = ["Proposed Model (GRPO + Linear SIL)", "Temp Spiking Decay"]
    colors = ['#1f77b4', '#d62728']
    
    for run, col in zip(runs, colors):
        ep, mins, means = load_eval_data(run)
        dyn = load_dynamics_data(run)
        
        if ep is not None and dyn is not None:
            final_min = mins[-1]
            # Plot Minimum (Solid) and Mean (Dashed) to show Confidence Gap
            ax1.plot(ep, mins, color=col, lw=2.5, label=f"Min E: {run} ({final_min:.3f})")
            ax1.plot(ep, means, color=col, lw=1.0, linestyle='--', alpha=0.6, label=f"Mean E: {run}")
            
            # Bottom Plot: Temperature
            ax2.plot(dyn['Epoch'], dyn['Temperature'], color=col, lw=1.5, label=f"Temp: {run}")

    ax1.set_title("Energy Evolution vs. Temperature Scheduling")
    ax1.set_ylabel("Evaluation Energy")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    
    ax2.set_ylabel("Exploration Temp")
    ax2.set_xlabel("Training Epochs")
    ax2.legend(loc="upper right", fontsize=8)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_temperature_dynamics_split.png"), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    print("Generating report plots...")
    plot_baseline_comparison()
    plot_architectural_ablations()
    plot_stability_kldiv()
    plot_temperature_dynamics()
    print(f"Plots saved to {OUTPUT_DIR}/")