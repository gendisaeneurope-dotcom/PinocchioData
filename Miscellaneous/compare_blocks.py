import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

block_data = 'Data_1/Subject3'
comparison_output = 'Blocks_Comparison'
os.makedirs(comparison_output, exist_ok=True)

joint_model = [
    'Single_leg_ankle_abd',
    'Single_leg_hip_abd',
    'Single_leg_hip_flex',
    'Single_leg_ankle_flex'
]

perturbation_blocks = ['Block_1000.0', 'Block_2000.0', 'Block_3000.0', 'Block_4000.0', 'Block_5000.0']

def load_trial(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=joint_model, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['time'] = df['marker_timestamp'] - df['marker_timestamp'].iloc[0]
    df[joint_model] = np.deg2rad(df[joint_model].astype(float))
    return df

# Find all Block 0 trials
block0_files = sorted([f for f in os.listdir(block_data) if f.startswith('Block_0.0')])

# Find all perturbation trials (blocks 1-4), skip catch trials
perturb_with_force = []
for f in sorted(os.listdir(block_data)):
    if not any(f.startswith(b) for b in perturbation_blocks):
        continue
    df_tmp = pd.read_csv(f'{block_data}/{f}')
    is_catch = df_tmp['motor_force'].abs().max() == 0
    print(f"{f}: {'CATCH — skip' if is_catch else 'keeping'}")
    if not is_catch:
        perturb_with_force.append(f)

print(f"\nBlock 0 trials: {len(block0_files)}")
print(f"Perturbation trials with force: {len(perturb_with_force)}")

# Compare Block 0 vs each perturbation block trial by trial
for b0_file in block0_files:
    trial_id = b0_file.replace('Block_0.0_trial_', '')   # e.g. '1.0.csv'

    for block_name in perturbation_blocks:
        b1_file = f'{block_name}_trial_{trial_id}'

        if b1_file not in perturb_with_force:
            continue

        df_b0 = load_trial(f'{block_data}/{b0_file}')
        df_b1 = load_trial(f'{block_data}/{b1_file}')

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()

        for i, col in enumerate(joint_model):
            axes[i].plot(df_b0['time'], df_b0[col], label='Block 0', color='blue')
            axes[i].plot(df_b1['time'], df_b1[col], label=block_name, color='red')
            axes[i].set_title(col)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Angle (rad)')
            axes[i].legend(fontsize=7)

        trial_name = trial_id.replace('.csv', '')
        plt.suptitle(f'Block 0 vs {block_name} — Trial {trial_name}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_output,
            f'compare_{block_name}_trial_{trial_name}.png'), dpi=100)
        plt.close()
        print(f"Saved: compare_{block_name}_trial_{trial_name}.png")

print("=== All comparisons done ===")
