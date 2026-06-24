# =====================================================
# Phase 0 — Model Validation & Data Exploration
# Load URDF, compute dynamics, plot experimental data.
# NO controller, NO PD, NO com_error.
# =====================================================
print("=== Script started ===")
import os, gc, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pinocchio as pin
from scipy.signal import butter, filtfilt
from pathlib import Path
print("=== Imports done ===")



# ── Configuration ─────────────────────────────────────────────────────────────
PLOT_DIR         = 'Plots'
TRIAL_DIR        = 'Data_1/trials'
CSV_RAW          = 'Data_1/resynchronized_data_subject003.csv'
CSV_OUT          = 'Data_1/Subject3_features.csv'
URDF_NAME        = 'Script/subject3_single_leg_4dof.urdf'
TRIAL_MIN_FRAMES = 50
FORCE_THRESHOLD  = 20.0  # N


JOINT_MODEL = [
    'Single_leg_ankle_abd',
    'Single_leg_hip_abd',
    'Single_leg_hip_flex',
    'Single_leg_ankle_flex',
]

os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(TRIAL_DIR, exist_ok=True)


# =====================================================
# 1. Build Pinocchio model
# =====================================================
mesh_dir  = os.getcwd()
urdf_path = os.path.join(mesh_dir, URDF_NAME)
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, mesh_dir)
data = model.createData()

q0 = pin.neutral(model)
v0 = np.zeros(model.nv)
a0 = np.zeros(model.nv)

total_mass = sum(model.inertias[i].mass for i in range(model.njoints))
com0       = pin.centerOfMass(model, data, q0)
M_mat      = pin.crba(model, data, q0)
h          = pin.rnea(model, data, q0, v0, a0)
tau_eom    = M_mat @ a0 + h

pin.computeJointJacobians(model, data, q0)
J = pin.getJointJacobian(
    model, data, model.njoints - 1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)

parent_id  = model.getJointId("hip_c_rotation1")
pin.forwardKinematics(model, data, q0)
com_pos    = pin.centerOfMass(model, data, q0)
parent_pos = data.oMi[parent_id].translation
r          = com_pos - parent_pos
F_test     = np.array([0.0, 0.0, 10.0])
M_cross    = np.cross(r, F_test)
R          = data.oMi[parent_id].rotation
fext_test  = [pin.Force.Zero() for _ in range(model.njoints)]
fext_test[parent_id] = pin.Force(R.T @ F_test, R.T @ M_cross)
tau_with_fext = pin.rnea(model, data, q0, v0, a0, fext_test)

with open("model_info.txt", "w") as f:
    f.write(f"Total mass: {total_mass}\n")
    f.write(f"CoM at neutral: {com0}\n")
    f.write("Joint limits:\n")
    for i in range(1, model.njoints):
        lo = model.lowerPositionLimit[i - 1]
        hi = model.upperPositionLimit[i - 1]
        f.write(f"  {model.names[i]}: [{lo:.3f}, {hi:.3f}] rad\n")
    f.write(f"Gravity: {model.gravity.linear}\n")
    f.write(f"Jacobian (last joint):\n{J}\n")
    f.write(f"Mass matrix M(q):\n{M_mat}\n")
    f.write(f"h(q, qdot): {h}\n")
    f.write(f"tau_eom: {tau_eom}\n")
    f.write(f"tau_with_fext: {tau_with_fext}\n")
print("Saved → model_info.txt")


# =====================================================
# 2. Load CSV + Dynamics Loop
# =====================================================
df_raw = pd.read_csv(CSV_RAW)

df_raw['motor_force'] = df_raw['motor_force'] / 1000.0
df_raw[JOINT_MODEL]   = np.deg2rad(df_raw[JOINT_MODEL].astype(float))
df_raw = df_raw.dropna(subset=JOINT_MODEL).reset_index(drop=True)
df_raw['time'] = df_raw['marker_timestamp'] - df_raw['marker_timestamp'].iloc[0]

dt_mean   = df_raw['time'].diff().mean()
nyquist   = 1 / (2 * dt_mean)
b, a_filt = butter(4, 3 / nyquist, btype='low')

df_filter = pd.DataFrame({'time': df_raw['time'].values})
for col in JOINT_MODEL:
    df_filter[col] = filtfilt(b, a_filt, df_raw[col].astype(float))

dt = df_filter['time'].diff()
for col in JOINT_MODEL:
    vel = (df_filter[col].diff() / dt).replace([np.inf, -np.inf], 0).fillna(0)
    acc = (vel.diff()            / dt).replace([np.inf, -np.inf], 0).fillna(0)
    df_filter[f'vel_{col}'] = vel
    df_filter[f'acc_{col}'] = acc

df_filter['motor_force'] = df_raw['motor_force'].values
df_filter['block_id']    = (df_raw['block_idx'] / 1000).round() * 1000

# ── Main dynamics loop ────────────────────────────────────────────────────
tau_buf = {col: [] for col in JOINT_MODEL}
com_buf = {'com_x': [], 'com_y': [], 'com_z': []}

for idx, row in df_filter.iterrows():
    if idx % 5000 == 0:
        print(f"  Frame {idx}/{len(df_filter)}...")
    q   = row[JOINT_MODEL].to_numpy(dtype=np.float64)
    vel = row[[f'vel_{c}' for c in JOINT_MODEL]].to_numpy(dtype=np.float64)
    acc = row[[f'acc_{c}' for c in JOINT_MODEL]].to_numpy(dtype=np.float64)
    tau = pin.rnea(model, data, q, vel, acc)
    for col in JOINT_MODEL:
        tau_buf[col].append(tau[JOINT_MODEL.index(col)])
    com = pin.centerOfMass(model, data, q)
    com_buf['com_x'].append(com[0])
    com_buf['com_y'].append(com[1])
    com_buf['com_z'].append(com[2])

print("  Main loop done.")

for col in JOINT_MODEL:
    df_filter[f'tau_{col}'] = tau_buf[col]
for key, vals in com_buf.items():
    df_filter[key] = vals

# ── tau_fext + tau_Jt on perturbed frames only ────────────────────────────
for col in JOINT_MODEL:
    df_filter[f'tau_fext_{col}'] = 0.0
    df_filter[f'tau_Jt_{col}']   = 0.0

perturbed_idx = df_filter[df_filter['motor_force'].abs() > 0.1].index
print(f"  tau_fext for {len(perturbed_idx)} perturbed frames...")

for count, idx in enumerate(perturbed_idx):
    if count % 2000 == 0:
        print(f"  Perturbed {count}/{len(perturbed_idx)}...")
    row     = df_filter.loc[idx]
    q       = row[JOINT_MODEL].to_numpy(dtype=np.float64)
    vel     = row[[f'vel_{c}' for c in JOINT_MODEL]].to_numpy(dtype=np.float64)
    acc     = row[[f'acc_{c}' for c in JOINT_MODEL]].to_numpy(dtype=np.float64)
    motor_f = float(row['motor_force'])
    pin.forwardKinematics(model, data, q)
    com_pos = pin.centerOfMass(model, data, q)
    jnt_pos = data.oMi[parent_id].translation
    jnt_rot = data.oMi[parent_id].rotation
    r       = com_pos - jnt_pos
    F       = np.array([0.0, motor_f, 0.0])
    fext    = [pin.Force.Zero() for _ in range(model.njoints)]
    fext[parent_id] = pin.Force(jnt_rot.T @ F, jnt_rot.T @ np.cross(r, F))
    tau_fext = pin.rnea(model, data, q, vel, acc, fext)
    pin.computeJointJacobians(model, data, q)
    J_com  = pin.jacobianCenterOfMass(model, data, q)
    tau_jt = J_com.T @ F
    for col in JOINT_MODEL:
        i = JOINT_MODEL.index(col)
        df_filter.loc[idx, f'tau_fext_{col}'] = tau_fext[i]
        df_filter.loc[idx, f'tau_Jt_{col}']   = tau_jt[i]

print("  tau_fext + tau_Jt done.")

out_cols = (
    ['time', 'block_id', 'motor_force']
    + JOINT_MODEL
    + [f'vel_{c}' for c in JOINT_MODEL]
    + [f'tau_{c}' for c in JOINT_MODEL]
    + [f'tau_fext_{c}' for c in JOINT_MODEL]
    + [f'tau_Jt_{c}' for c in JOINT_MODEL]
    + ['com_x', 'com_y', 'com_z']
)
df_filter[out_cols].to_csv(CSV_OUT, index=False)
print(f"Saved → {CSV_OUT}")


# =====================================================
# 3. Overview Plots + Per-Trial Plots
# =====================================================
def save_overview(cols, title, filename):
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in cols:
        ax.plot(df_filter['time'].values[::5], df_filter[col].values[::5],
                label=col, rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=100)
    plt.close(fig)

save_overview(JOINT_MODEL,                        "Joint Angles (rad)",         "plot_angles.png")
save_overview([f'vel_{c}' for c in JOINT_MODEL],  "Joint Velocities (rad/s)",   "plot_velocities.png")
save_overview([f'acc_{c}' for c in JOINT_MODEL],  "Joint Accelerations (r/s²)", "plot_accelerations.png")
save_overview([f'tau_{c}' for c in JOINT_MODEL],  "Joint Torques (Nm)",         "plot_torques.png")
save_overview(['com_x', 'com_y', 'com_z'],         "CoM Position (m)",           "plot_CoM.png")
print("=== Overview plots saved ===")

# ── Trial detection ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_OUT)

print("\nmotor_force stats:")
print(df['motor_force'].describe())
for thresh in [1, 5, 10, 20, 30]:
    n = (df['motor_force'].abs() > thresh).sum()
    print(f"  Frames above {thresh}N: {n}")

df['is_perturbed'] = (df['motor_force'].abs() > FORCE_THRESHOLD).astype(int)
df['trial_id']     = (df['is_perturbed'].diff() == 1).cumsum()
df.loc[df['is_perturbed'] == 0, 'trial_id'] = -1

trial_lengths = df[df['trial_id'] > 0].groupby('trial_id').size()
valid_trials  = trial_lengths[trial_lengths >= TRIAL_MIN_FRAMES].index
print(f"\nTrial length stats:\n{trial_lengths.describe()}")
print(f"Valid trials: {len(valid_trials)}")

block0 = df[df['block_id'] == 0]
print(f"\nBlock 0 max |motor_force|: {block0['motor_force'].abs().max():.4f} N")
print(f"Block 0 perturbed frames:   {block0['is_perturbed'].sum()}")

# ── Block-level force diagnostic ──────────────────────────────────────────
print("\n=== Block Force Summary ===")
for block_id in sorted(df['block_id'].unique()):
    block       = df[df['block_id'] == block_id]
    max_f       = block['motor_force'].abs().max()
    n_perturbed = (block['motor_force'].abs() > FORCE_THRESHOLD).sum()
    duration    = block['time'].max() - block['time'].min()
    status      = "NO FORCE ✓" if n_perturbed == 0 else f"{n_perturbed} perturbed frames"
    print(f"  Block {int(block_id):5d} | {duration:6.0f}s | max: {max_f:6.2f}N | {status}")

quiet = df[df['is_perturbed'] == 0]
noisy = df[df['is_perturbed'] == 1]
print(f"\nTotal quiet frames:     {len(quiet)} ({len(quiet)/len(df)*100:.1f}%)")
print(f"Total perturbed frames: {len(noisy)} ({len(noisy)/len(df)*100:.1f}%)")

first_perturbed_time = df[df['is_perturbed'] == 1]['time'].min()
last_quiet_frame     = df[(df['is_perturbed'] == 0) &
                          (df['time'] < first_perturbed_time)].iloc[-1]
print(f"\nFirst perturbation at:  {first_perturbed_time:.2f}s")
print(f"Last quiet frame at:    {last_quiet_frame['time']:.2f}s  "
      f"(block {int(last_quiet_frame['block_id'])})")

# ── Force overview plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(df['time'], df['motor_force'], color='crimson', lw=0.6, rasterized=True)
ax.axhline( FORCE_THRESHOLD, color='k', ls='--', lw=0.8, label=f'+{FORCE_THRESHOLD}N threshold')
ax.axhline(-FORCE_THRESHOLD, color='k', ls='--', lw=0.8)
ax.axvline(first_perturbed_time, color='orange', lw=1.2, label='First perturbation')
for i, block_id in enumerate(sorted(df['block_id'].unique())):
    block = df[df['block_id'] == block_id]
    if i % 2 == 0:
        ax.axvspan(block['time'].min(), block['time'].max(), alpha=0.08, color='steelblue')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Motor Force [N]')
ax.set_title('Full Recording — Motor Force with Block Boundaries')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'force_overview.png'), dpi=120)
plt.close()
print("Saved → force_overview.png")

removed = glob.glob(f'{TRIAL_DIR}/*.png')
for f in removed:
    os.remove(f)
print(f"\nCleaned {len(removed)} old trial plots")

JOINT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# ── Block 0 diagnostics ───────────────────────────────────────────────────
print("\n=== Block 0 Diagnostics ===")
print(df[df['block_id'] == 0]['motor_force'].abs().describe())
print(f"\nBlock 0 duration: {df[df['block_id'] == 0]['time'].max():.1f}s")
print(f"\nFrames per block:")
print(df['block_id'].value_counts().sort_index())

# ── Save Block 0 as trial_000 ─────────────────────────────────────────────
block0_trial = df[df['block_id'] == 0].reset_index(drop=True)
t0           = block0_trial['time'] - block0_trial['time'].iloc[0]

fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
fig.suptitle('Trial 000 — Baseline (Block 0, no perturbation)', fontsize=13)
axes[0].plot(t0, block0_trial['motor_force'], color='crimson')
axes[0].set_ylabel('Motor Force [N]')
axes[0].axhline(0, color='k', lw=0.6, ls='--')
for col, clr in zip(JOINT_MODEL, JOINT_COLORS):
    axes[1].plot(t0, block0_trial[col], label=col.replace('Single_leg_', ''), color=clr)
axes[1].set_ylabel('Joint Angles [rad]')
axes[1].legend(fontsize=6, ncol=2)
for col, clr in zip(JOINT_MODEL, JOINT_COLORS):
    axes[2].plot(t0, block0_trial[f'tau_{col}'], label=col.replace('Single_leg_', ''), color=clr)
axes[2].set_ylabel('Joint Torques [Nm]')
axes[2].legend(fontsize=6, ncol=2)
axes[2].axhline(0, color='k', lw=0.6, ls='--')
axes[3].plot(t0, block0_trial['com_x'], label='CoM X', color='steelblue')
axes[3].plot(t0, block0_trial['com_y'], label='CoM Y', color='darkorange')
axes[3].set_ylabel('CoM X/Y [m]')
axes[3].legend(fontsize=7)
axes[3].axhline(0, color='k', lw=0.6, ls='--')
axes[4].plot(t0, block0_trial['com_z'], color='mediumpurple')
axes[4].set_ylabel('CoM Z [m]')
axes[4].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig(f'{TRIAL_DIR}/trial_000.png', dpi=120)
plt.close()
print("Saved → trial_000.png (Baseline)")

# ── Per-trial 5-panel plots ───────────────────────────────────────────────
for tid in valid_trials:
    trial = df[df['trial_id'] == tid].reset_index(drop=True)
    t     = trial['time'] - trial['time'].iloc[0]
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Trial {int(tid):03d}', fontsize=13)
    axes[0].plot(t, trial['motor_force'], color='crimson')
    axes[0].set_ylabel('Motor Force [N]')
    axes[0].axhline(0, color='k', lw=0.6, ls='--')
    for col, clr in zip(JOINT_MODEL, JOINT_COLORS):
        axes[1].plot(t, trial[col], label=col.replace('Single_leg_', ''), color=clr)
    axes[1].set_ylabel('Joint Angles [rad]')
    axes[1].legend(fontsize=6, ncol=2)
    for col, clr in zip(JOINT_MODEL, JOINT_COLORS):
        axes[2].plot(t, trial[f'tau_{col}'], label=col.replace('Single_leg_', ''), color=clr)
    axes[2].set_ylabel('Joint Torques [Nm]')
    axes[2].legend(fontsize=6, ncol=2)
    axes[2].axhline(0, color='k', lw=0.6, ls='--')
    axes[3].plot(t, trial['com_x'], label='CoM X', color='steelblue')
    axes[3].plot(t, trial['com_y'], label='CoM Y', color='darkorange')
    axes[3].set_ylabel('CoM X/Y [m]')
    axes[3].legend(fontsize=7)
    axes[3].axhline(0, color='k', lw=0.6, ls='--')
    axes[4].plot(t, trial['com_z'], color='mediumpurple')
    axes[4].set_ylabel('CoM Z [m]')
    axes[4].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(f'{TRIAL_DIR}/trial_{int(tid):03d}.png', dpi=120)
    plt.close()

print(f"Saved {len(valid_trials)} trial plots → {TRIAL_DIR}/")

# ── Torque comparison: perturbed vs unperturbed ───────────────────────────
block0_frames         = df[df['block_id'] == 0].copy().reset_index(drop=True)
median_len            = int(trial_lengths.median())
n_unperturbed_windows = len(block0_frames) // median_len
n_perturbed_trials    = len(valid_trials)

unperturbed_segs = {col: [] for col in JOINT_MODEL}
for i in range(0, len(block0_frames) - median_len, median_len):
    seg = block0_frames.iloc[i:i + median_len]
    for col in JOINT_MODEL:
        unperturbed_segs[col].append(seg[f'tau_{col}'].values)

perturbed_segs = {col: [] for col in JOINT_MODEL}
for tid in valid_trials:
    seg = df[df['trial_id'] == tid].reset_index(drop=True)
    for col in JOINT_MODEL:
        vals   = seg[f'tau_{col}'].values
        padded = np.pad(vals, (0, max(0, median_len - len(vals))),
                        constant_values=np.nan)[:median_len]
        perturbed_segs[col].append(padded)

t_common = np.arange(median_len) / 100.0

fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
fig.suptitle(
    f'Joint Torques — Perturbed vs Unperturbed\n'
    f'Unperturbed: Block 0 ({n_unperturbed_windows} windows) | '
    f'Perturbed: all valid trials (n={n_perturbed_trials})',
    fontsize=11
)
for i, (col, clr) in enumerate(zip(JOINT_MODEL, JOINT_COLORS)):
    short    = col.replace('Single_leg_', '')
    unp_arr  = np.array(unperturbed_segs[col])
    unp_mean = np.nanmean(unp_arr, axis=0)
    unp_std  = np.nanstd(unp_arr,  axis=0)
    per_arr  = np.array(perturbed_segs[col])
    per_mean = np.nanmean(per_arr, axis=0)
    per_std  = np.nanstd(per_arr,  axis=0)
    axes[i].plot(t_common, unp_mean, color='gray', lw=1.8, ls='--',
                 label=f'Unperturbed mean (n={n_unperturbed_windows})')
    axes[i].fill_between(t_common, unp_mean - unp_std, unp_mean + unp_std,
                         alpha=0.20, color='gray', label='Unperturbed ±1 std')
    axes[i].plot(t_common, per_mean, color=clr, lw=1.8,
                 label=f'Perturbed mean (n={n_perturbed_trials})')
    axes[i].fill_between(t_common, per_mean - per_std, per_mean + per_std,
                         alpha=0.25, color=clr, label='Perturbed ±1 std')
    axes[i].axhline(0, color='k', lw=0.6, ls=':')
    axes[i].set_ylabel(f'{short} [Nm]')
    axes[i].legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Time within trial [s]')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'torque_comparison.png'), dpi=120)
plt.close()
print("Saved → torque_comparison.png")


# =====================================================
# 4. Block-average plots (mean ± std per block, baseline-corrected)
# =====================================================
BLOCK_MAP = {
    'Baseline':    ([0],          'steelblue'),
    'Early Adapt': ([1000, 2000], 'darkorange'),
    'Late Adapt':  ([3000, 4000], 'crimson'),
    'Washout':     ([5000],       'seagreen'),
}

block_trial_counts = {}
for label, (block_ids, _) in BLOCK_MAP.items():
    if 0 in block_ids or 5000 in block_ids:
        b = df[df['block_id'].isin(block_ids)]
        block_trial_counts[label] = len(b) // int(trial_lengths.median())
    else:
        tids = df[(df['block_id'].isin(block_ids)) &
                  (df['is_perturbed'] == 1)]['trial_id'].unique()
        block_trial_counts[label] = len([t for t in tids if t in valid_trials])
print("Block trial counts:", block_trial_counts)


PRE_TRIAL_FRAMES = 20

def block_avg_plot(df_src, y_col, ylabel, filename, valid_trial_ids, trim_len):
    fig, ax = plt.subplots(figsize=(10, 4))
    title_parts = []

    for label, (block_ids, color) in BLOCK_MAP.items():
        if 5000 in block_ids:
            continue

        block = df_src[df_src['block_id'].isin(block_ids)]

        if 0 in block_ids:
            ref_mean = np.nanmean(block[y_col].values)
            ref_std  = np.nanstd(block[y_col].values)
            ax.axhline(ref_mean, color=color, lw=1.5, ls='--', label=f'{label} (ref)')
            ax.axhspan(ref_mean - ref_std, ref_mean + ref_std, alpha=0.08, color=color)
            title_parts.append(f'{label} (ref)')
            continue

        trial_ids = block[block['is_perturbed'] == 1]['trial_id'].unique()
        trial_ids = [tid for tid in trial_ids if tid in valid_trial_ids]
        segments  = []

        for tid in trial_ids:
            trial_start_idx = df_src[df_src['trial_id'] == tid].index[0]
            pre_frames = df_src.loc[
                max(0, trial_start_idx - PRE_TRIAL_FRAMES):trial_start_idx - 1,
                y_col
            ].values
            pre_mean = np.nanmean(pre_frames) if len(pre_frames) > 0 else 0.0
            seg_data = df_src[df_src['trial_id'] == tid][y_col].values
            seg_data = seg_data - pre_mean
            segments.append(seg_data)

        n = len(segments)
        title_parts.append(
            f'{label} (n={n}, tids {int(trial_ids[0]):03d}–{int(trial_ids[-1]):03d})'
            if trial_ids else f'{label} (n=0)'
        )

        if not segments:
            continue

        trim_len_actual = min(len(s) for s in segments)
        trim_len_actual = min(trim_len, trim_len_actual)
        padded = np.array([s[:trim_len_actual] for s in segments])
        mean   = np.nanmean(padded, axis=0)
        std    = np.nanstd(padded,  axis=0)
        x      = np.arange(trim_len_actual) / 100.0

        ax.plot(x, mean, label=f'{label} (n={n})', color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

    ax.axhline(0, color='k', lw=0.6, ls=':', label='Pre-trial baseline')
    ax.set_title(f'Block average ± std — pre-trial corrected: {ylabel}\n'
                 + ' | '.join(title_parts), fontsize=9)
    ax.set_xlabel('Time within trial [s]')
    ax.set_ylabel(f'Δ {ylabel}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=120)
    plt.close(fig)
    print(f"Saved → {filename}")


# Compute trim_len once here, outside the function
lengths_all = [len(df[df['trial_id'] == tid]) for tid in valid_trials]
print(f"Shortest trial: {min(lengths_all)} frames = {min(lengths_all)/100:.2f}s")
print(f"Median trial:   {int(np.median(lengths_all))} frames = {int(np.median(lengths_all))/100:.2f}s")
print(f"Longest trial:  {max(lengths_all)} frames = {max(lengths_all)/100:.2f}s")
TRIM_LEN = int(np.percentile(lengths_all, 10))
print(f"trim_len → {TRIM_LEN} frames = {TRIM_LEN/100:.2f}s")

block_avg_plot(df, 'com_y', 'CoM Y [m]', 'block_avg_com_y.png', valid_trials, TRIM_LEN)
block_avg_plot(df, 'com_x', 'CoM X [m]', 'block_avg_com_x.png', valid_trials, TRIM_LEN)
for col in JOINT_MODEL:
    short = col.replace('Single_leg_', '')
    block_avg_plot(df, col,          f'{short} [rad]', f'block_avg_{short}.png',     valid_trials, TRIM_LEN)
    block_avg_plot(df, f'tau_{col}', f'{short} [Nm]',  f'block_avg_tau_{short}.png', valid_trials, TRIM_LEN)

print("=== All block-average plots saved ===")
print("MeshCat skipped. Script finished.")