# Phase 0 #
# Objectives: 
# 1. Model Validations: Load URDF/XML in Pinocchio, print total mass, compute CoM at neutral pose, verify joint limits, check gravity vector, compute Jacobians
# 2. Compute M(q) with crba, h(q,q̇) with rnea, implement equation of motion, add external force at CoM

# =====================================================
# 0. Imports
# =====================================================
print("=== Script started ===")
import os, signal
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
import time as time_module
from scipy.signal import butter, filtfilt
import gc
import glob
print("=== Imports done ===")

root = "world"
plot = True
figure_output = 'Plots'
os.makedirs(figure_output, exist_ok=True)

joint_model = [
    'Single_leg_ankle_abd',
    'Single_leg_hip_abd',
    'Single_leg_hip_flex',
    'Single_leg_ankle_flex'
]

# =====================================================
# 1. Build Pinocchio model
# Load URDF, extract static quantities: mass, CoM,
# joint limits, gravity, Jacobian, M(q), h, tau_eom,
# external force at CoM. Save to model_info.txt.
# =====================================================
mesh_dir = os.path.dirname(os.path.abspath(__file__))
urdf_filename = "subject3_single_leg_4dof.urdf"
urdf_model_path = os.path.join(mesh_dir, urdf_filename)

model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)
data = model.createData()

q0 = pin.neutral(model)
v0 = np.zeros(model.nv)
a0 = np.zeros(model.nv)

total_mass = sum([model.inertias[i].mass for i in range(model.njoints)])
com0       = pin.centerOfMass(model, data, q0)

for i in range(1, model.njoints):
    name  = model.names[i]
    lower = model.lowerPositionLimit[i-1]
    upper = model.upperPositionLimit[i-1]
    print(f"Joint {name}: [{lower:.3f}, {upper:.3f}] rad")

print(model.gravity.linear)

pin.computeJointJacobians(model, data, q0)
J = pin.getJointJacobian(model, data, model.njoints-1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

M_mat   = pin.crba(model, data, q0)
h       = pin.rnea(model, data, q0, v0, a0)
tau_eom = M_mat @ a0 + h

parent_id  = model.getJointId("hip_c_rotation1")
pin.forwardKinematics(model, data, q0)
com_pos    = pin.centerOfMass(model, data, q0)
parent_pos = data.oMi[parent_id].translation
r          = com_pos - parent_pos
F          = np.array([0.0, 0.0, 10.0])
M_cross    = np.cross(r, F)
R          = data.oMi[parent_id].rotation
fext       = [pin.Force.Zero() for _ in range(model.njoints)]
fext[parent_id] = pin.Force(R.T @ F, R.T @ M_cross)
tau_with_fext = pin.rnea(model, data, q0, v0, a0, fext)

with open("model_info.txt", "w") as f:
    f.write(f"Total mass: {total_mass}\n")
    f.write(f"CoM at neutral: {com0}\n")
    f.write("Joint limits:\n")
    for i in range(1, model.njoints):
        name  = model.names[i]
        lower = model.lowerPositionLimit[i-1]
        upper = model.upperPositionLimit[i-1]
        f.write(f"  Joint {name}: [{lower:.3f}, {upper:.3f}] rad\n")
    f.write(f"Gravity: {model.gravity.linear}\n")
    f.write(f"Jacobian of last joint:\n{J}\n")
    f.write(f"Mass matrix M(q):\n{M_mat}\n")
    f.write(f"h(q, qdot): {h}\n")
    f.write(f"tau_eom: {tau_eom}\n")
    f.write(f"tau_with_fext: {tau_with_fext}\n")
print("Saved → model_info.txt")


# =====================================================
# 2. Load CSV + Dynamics Loop
# Load raw CSV, filter joint angles, compute
# velocities/accelerations, run RNEA per frame,
# compute CoM error. tau_fext and tau_Jt computed
# separately only on perturbed frames.
# Save to Subject3_features.csv.
# =====================================================
csv_files = 'Data_1/resynchronized_data_subject003.csv'
df = pd.read_csv(csv_files)
df[joint_model] = np.deg2rad(df[joint_model].astype(float))
df = df.dropna(subset=joint_model).reset_index(drop=True)
df['time'] = df['marker_timestamp'] - df['marker_timestamp'].iloc[0]

df_filtered = pd.DataFrame()
df_filtered['time'] = df['time'].values
for col in joint_model:
    df_filtered[col] = df[col].values

cutoff_freq       = 3
nyquist_freq      = 1 / (2 * df['time'].diff().mean())
normalized_cutoff = cutoff_freq / nyquist_freq
b, a_filt = butter(4, normalized_cutoff, btype='low')
for col in joint_model:
    df_filtered[col] = filtfilt(b, a_filt, df_filtered[col].astype(float))

for col in joint_model:
    df_filtered['vel_' + col] = df_filtered[col].diff() / df_filtered['time'].diff()
    df_filtered['acc_' + col] = df_filtered['vel_' + col].diff() / df_filtered['time'].diff()
    df_filtered['tau_' + col] = 0.0
    df_filtered[col]          = df_filtered[col].fillna(0)
    df_filtered['vel_' + col] = df_filtered['vel_' + col].replace([np.inf, -np.inf], 0).fillna(0)
    df_filtered['acc_' + col] = df_filtered['acc_' + col].replace([np.inf, -np.inf], 0).fillna(0)

df_filtered['com_x'] = 0.0
df_filtered['com_y'] = 0.0
df_filtered['com_z'] = 0.0
df_filtered['com_error_x'] = 0.0
df_filtered['com_error_y'] = 0.0
df_filtered['com_error_z'] = 0.0

# ── Main dynamics loop — RNEA + CoM only, no tau_fext ──────────────────
tau_results      = {col: [] for col in joint_model}
tau_fext_results = {col: np.zeros(len(df_filtered)) for col in joint_model}
com_x, com_y, com_z = [], [], []
com_error_x, com_error_y, com_error_z = [], [], []

for idx, d_i in df_filtered.iterrows():
    if idx % 5000 == 0:
        print(f"  Processing frame {idx}/{len(df_filtered)}...")

    q_exp         = d_i[joint_model].to_numpy(dtype=np.float64)
    velocities    = d_i[['vel_' + col for col in joint_model]].to_numpy(dtype=np.float64)
    accelerations = d_i[['acc_' + col for col in joint_model]].to_numpy(dtype=np.float64)

    # RNEA — always
    tau = pin.rnea(model, data, q_exp, velocities, accelerations)
    for col in joint_model:
        tau_results[col].append(tau[joint_model.index(col)])

    # CoM — always
    com = pin.centerOfMass(model, data, q_exp)
    com_x.append(com[0]); com_y.append(com[1]); com_z.append(com[2])

    com_desired = np.array([0.0, 0.0, com[2]])
    com_error   = com_desired - com
    com_error_x.append(com_error[0])
    com_error_y.append(com_error[1])
    com_error_z.append(com_error[2])

    # tau_fext — skip here, filled zero, computed below

print("  Main loop done.")

# ── Bulk assign ─────────────────────────────────────────────────────────
for col in joint_model:
    df_filtered['tau_' + col] = tau_results[col]
df_filtered['com_x'] = com_x
df_filtered['com_y'] = com_y
df_filtered['com_z'] = com_z
df_filtered['com_error_x'] = com_error_x
df_filtered['com_error_y'] = com_error_y
df_filtered['com_error_z'] = com_error_z
df_filtered['block_id']    = df['block_idx'].round(-3).values
df_filtered['motor_force'] = df['motor_force'].values

# ── Build df_out ─────────────────────────────────────────────────────────
df_out = pd.DataFrame({
    'time':        df_filtered['time'].values,
    'block_id':    df_filtered['block_id'].values,
    'com_x':       com_x, 'com_y': com_y, 'com_z': com_z,
    'com_error_x': com_error_x,
    'com_error_y': com_error_y,
    'com_error_z': com_error_z,
    'motor_force': df_filtered['motor_force'].values
})
for col in joint_model:
    df_out['tau_' + col]      = tau_results[col]
    df_out['tau_fext_' + col] = 0.0
    df_out['tau_Jt_' + col]   = 0.0

# ── mN → N conversion ───────────────────────────────────────────────────
df['motor_force']          = df['motor_force'] / 1000.0
df_filtered['motor_force'] = df_filtered['motor_force'] / 1000.0
df_out['motor_force']      = df_out['motor_force'] / 1000.0

# ── tau_fext + tau_Jt — perturbed frames only ───────────────────────────
perturbed_mask = df['motor_force'].abs() > 0.1
perturbed_idx  = df_filtered[perturbed_mask.values].index
print(f"  Computing tau_fext + tau_Jt for {len(perturbed_idx)} perturbed frames...")

for count, idx in enumerate(perturbed_idx):
    if count % 2000 == 0:
        print(f"  Perturbed frame {count}/{len(perturbed_idx)}...")

    q_exp         = df_filtered.loc[idx, joint_model].to_numpy(dtype=np.float64)
    motor_force   = float(df.loc[idx, 'motor_force'])
    velocities    = df_filtered.loc[idx, ['vel_' + col for col in joint_model]].to_numpy(dtype=np.float64)
    accelerations = df_filtered.loc[idx, ['acc_' + col for col in joint_model]].to_numpy(dtype=np.float64)

    pin.forwardKinematics(model, data, q_exp)
    com_pos   = pin.centerOfMass(model, data, q_exp)
    joint_pos = data.oMi[parent_id].translation
    joint_rot = data.oMi[parent_id].rotation
    r         = com_pos - joint_pos
    F         = np.array([0.0, motor_force, 0.0])
    M_cross   = np.cross(r, F)
    fext      = [pin.Force.Zero() for _ in range(model.njoints)]
    fext[parent_id] = pin.Force(joint_rot.T @ F, joint_rot.T @ M_cross)
    tau_fext  = pin.rnea(model, data, q_exp, velocities, accelerations, fext)

    pin.computeJointJacobians(model, data, q_exp)
    J_CoM        = pin.jacobianCenterOfMass(model, data, q_exp)
    tau_de_force = J_CoM.T @ F

    for col in joint_model:
        df_out.loc[idx, 'tau_fext_' + col] = tau_fext[joint_model.index(col)]
        df_out.loc[idx, 'tau_Jt_' + col]   = tau_de_force[joint_model.index(col)]

print("  tau_fext + tau_Jt done.")

# ── PD controller ────────────────────────────────────────────────────────
df_out['com_error_y_dot'] = df_out['com_error_y'].diff() / df_out['time'].diff()
df_out['com_error_y_dot'] = df_out['com_error_y_dot'].fillna(0)
df_out['tau_PD']          = 100.0 * df_out['com_error_y'] + 10.0 * df_out['com_error_y_dot']

df_out.to_csv('Data_1/Subject3_features.csv', index=False)
print("Saved → Data_1/Subject3_features.csv")


# =====================================================
# 3. Plots
# Save joint angle/velocity/acceleration/torque/CoM
# plots as PNG. Load features CSV, detect perturbation
# trials, save one 4-panel plot per trial.
# =====================================================
if plot:
    def save_fig(data_cols, title, filename):
        fig, ax = plt.subplots(figsize=(10, 4))
        for col in data_cols:
            ax.plot(df_filtered['time'].values[::5],
                    df_filtered[col].values[::5],
                    label=col, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(figure_output, filename), dpi=100)
        plt.close(fig)

    save_fig(joint_model,                        "Joint Angles (rad)",         "plot_angles.png")
    save_fig(['vel_' + c for c in joint_model],  "Joint Velocities (rad/s)",   "plot_velocities.png")
    save_fig(['acc_' + c for c in joint_model],  "Joint Accelerations (r/s²)", "plot_accelerations.png")
    save_fig(['tau_' + c for c in joint_model],  "Joint Torques (Nm)",         "plot_torques.png")
    save_fig(['com_x', 'com_y', 'com_z'],        "CoM Position (m)",           "plot_CoM.png")
    print("=== All plots saved ===")

# ── Trial detection with threshold ──────────────────────────────────────
df = pd.read_csv('Data_1/Subject3_features.csv')

# ── Check distribution to confirm threshold ──────────────────────────────
print("motor_force stats:")
print(df['motor_force'].describe())
print("Frames above 1N:",  (df['motor_force'].abs() > 1.0).sum())
print("Frames above 5N:",  (df['motor_force'].abs() > 5.0).sum())
print("Frames above 10N:", (df['motor_force'].abs() > 10.0).sum())
print("Frames above 20N:", (df['motor_force'].abs() > 20.0).sum())
print("Frames above 30N:", (df['motor_force'].abs() > 30.0).sum())

# ── Apply threshold ───────────────────────────────────────────────────────
df['is_perturbed'] = (df['motor_force'].abs() > 20.0).astype(int)
df['trial_id']     = (df['is_perturbed'].diff() == 1).cumsum()
df.loc[df['is_perturbed'] == 0, 'trial_id'] = -1

# ── check trial lengths before filtering ─────────────────
trial_lengths = df[df['trial_id'] > 0].groupby('trial_id').size()
print("Trial length stats:")
print(trial_lengths.describe())
print("Shortest trial:", trial_lengths.min(), "frames")
print("Longest trial:",  trial_lengths.max(), "frames")

# ── Remove micro-fragments shorter than 100 frames (1s) ──────────────────
trial_lengths = df[df['trial_id'] > 0].groupby('trial_id').size()
valid_trials  = trial_lengths[trial_lengths >= 50].index
trials        = valid_trials
print(f"Valid trials after duration filter: {len(trials)}")

os.makedirs('Data_1/trials', exist_ok=True)

# ── Clean old trial plots before saving new ones ────────────────────────
old_files = glob.glob('Data_1/trials/*.png')
for f in old_files:
    os.remove(f)
print(f"Cleaned {len(old_files)} old trial plots")

os.makedirs('Data_1/trials', exist_ok=True)

for tid in trials:
    trial = df[df['trial_id'] == tid].reset_index(drop=True)
    t     = trial['time'] - trial['time'].iloc[0]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f'Trial {int(tid)}', fontsize=13)
    axes[0].plot(t, trial['motor_force'],                  color='red')
    axes[0].set_ylabel('Motor Force [N]')
    axes[1].plot(t, trial['com_error_y'],                  color='orange')
    axes[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel('CoM Error Y [m]')
    axes[2].plot(t, trial['tau_Jt_Single_leg_hip_abd'],    color='blue')
    axes[2].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[2].set_ylabel('Hip Abd [Nm]')
    axes[3].plot(t, trial['tau_Jt_Single_leg_ankle_flex'], color='green')
    axes[3].axhline(0, color='k', linestyle='--', linewidth=0.8)
    axes[3].set_ylabel('Ankle Flex [Nm]')
    axes[3].set_xlabel('Time [s]')
    plt.tight_layout()
    plt.savefig(f'Data_1/trials/trial_{int(tid):03d}.png', dpi=120)
    plt.close()
print(f"Saved {len(trials)} trial plots → Data_1/trials/")


# =====================================================
# 4. Trajectory Analysis
# Reload raw CSV, recompute single-leg joint angles
# from bilateral markers, check time sync, detect
# trial boundaries, print diagnostic info.
# =====================================================
csv_path = 'Data_1/resynchronized_data_subject003.csv'
df = pd.read_csv(csv_path, low_memory=False)
df["t_sync"] = pd.to_numeric(df["t_sync"], errors="coerce")
df = df.dropna(subset=["t_sync"]).reset_index(drop=True)
df["t_sync"] = df["t_sync"] - df["t_sync"].iloc[0]

deg2rad = np.pi / 180.0
df['Single_leg_hip_abd']    = 0.5 * (df['hip_adduction_r'] - df['hip_adduction_l']) * deg2rad
df['Single_leg_hip_flex']   = 0.5 * (df['hip_flexion_r']   + df['hip_flexion_l'])   * deg2rad
df['Single_leg_ankle_flex'] = df['subtalar_sagittal_tilt_rad']
df['Single_leg_ankle_abd']  = df['subtalar_frontal_tilt_rad']

t_sync   = df["t_sync"].values
restarts = np.where(np.diff(t_sync) < 0)[0] + 1
print(f"Sampling rate: {1 / np.mean(np.diff(t_sync)):.1f} Hz")
print(f"Number of trials: {len(restarts) + 1}")


# =====================================================
# 5. MeshCat Visualization
# Animate skeleton frame by frame using filtered
# joint angles. Overlay red sphere at CoM position.
# =====================================================
# PREVIEW_FRAMES = 500

#viz = MeshcatVisualizer(model, collision_model, visual_model)
#viz.initViewer(zmq_url="tcp://127.0.0.1:6001", open=False)
#time_module.sleep(2)
#viz.loadViewerModel(rootNodeName=root)
#print("MeshCat open at: http://127.0.0.1:7001/static/")

#viz.display(pin.neutral(model))
#viz.viewer['com'].set_object(g.Sphere(0.08), g.MeshLambertMaterial(color=0xff0000))

#try:
#    for i in range(min(PREVIEW_FRAMES, len(df_filtered))):
#        q   = df_filtered[joint_model].iloc[i].values.astype(float)
#        viz.display(q)
#        com = df_filtered[['com_x', 'com_y', 'com_z']].iloc[i].values.astype(float)
#        viz.viewer['com'].set_transform(tf.translation_matrix(com))
#        time_module.sleep(1/60)
#finally:
#    proc = viz.viewer.window.server_proc
#    proc.terminate()
#    try:
#        proc.wait(timeout=3)
#    except Exception:
#        os.kill(proc.pid, signal.SIGTERM)
print("MeshCat stopped. Script finished.")