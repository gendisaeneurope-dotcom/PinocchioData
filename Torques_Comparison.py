import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

df_all = pd.read_csv('Data_1/Subject3_processed.csv')

comparison_output = 'Plots_Torque_Comparison_New'
os.makedirs(comparison_output, exist_ok=True)

torque_cols = [
    'tau_Single_leg_ankle_abd',
    'tau_Single_leg_hip_abd',
    'tau_Single_leg_hip_flex',
    'tau_Single_leg_ankle_flex'
]

df_b0 = df_all[df_all['block_id'] == 0.0]

perturbation_blocks = [b for b in df_all['block_id'].unique() if b != 0.0]

for block_id in sorted(perturbation_blocks):
    df_b0 = df_all[df_all['block_id'] == 0.0].copy()
    perturbation_blocks = df_all[(df_all['block_id'] == block_id) & (df_all['motor_force'] != 0)].copy()

    if perturbation_blocks.empty:
        print(f"Block {block_id}: no perturbation trials, skipping")
        continue

    # Reset time to 0 for each block
    df_b0['time'] = df_b0['time'] - df_b0['time'].iloc[0]
    perturbation_blocks['time'] = perturbation_blocks['time'] - perturbation_blocks['time'].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(torque_cols):
        axes[i].plot(df_b0['time'].values, df_b0[col].values, label='Block 0', color='blue', alpha=0.6)
        axes[i].plot(perturbation_blocks['time'].values, perturbation_blocks[col].values, label=f'Block {block_id}', color='red', alpha=0.6)
        axes[i].set_title(col)
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Torque (Nm)')
        axes[i].legend(fontsize=7)

    plt.suptitle(f'Block 0 vs Block {block_id} — Joint Torques', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_output, f'compare_torques_b0_vs_b{block_id}.png'), dpi=100)
    plt.close()
    print(f"Saved → compare_torques_b0_vs_b{block_id}.png")