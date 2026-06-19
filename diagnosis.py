print("=== Script started ===")

import os
print("os OK")
import sys
print("sys OK")
from pathlib import Path
print("pathlib OK")
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import rand
matplotlib.use("Agg")
print("matplotlib OK")
import numpy as np
print("numpy OK")
print("pyplot OK")
import pandas as pd
print("pandas OK")
import pinocchio as pin
print("pinocchio OK")
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
print("MeshcatVisualizer OK")
import time as time_module
print("time OK")
from scipy.signal import butter, filtfilt
print("scipy OK")
import gc
print("gc OK")

print("=== Imports done ===")

root = "world"
plot = True

all_joints = ['Single_leg_ankle_abd', # eversion/inversion    
            'Single_leg_hip_abd', # abduction/adduction
            'Single_leg_hip_flex', # flexion/extension
            'Single_leg_ankle_flex'] # plantar/dorsiflexion]

joint_model = [
            'Single_leg_ankle_abd', # eversion/inversion    
            'Single_leg_hip_abd', # abduction/adduction
            'Single_leg_hip_flex', # flexion/extension
            'Single_leg_ankle_flex' # plantar/dorsiflexion
            ]


# =====================================================
# 1. Build Pinocchio model
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
com0 = pin.centerOfMass(model, data, q0)

for i in range(1, model.njoints):
    name = model.names[i]
    lower = model.lowerPositionLimit[i-1]
    upper = model.upperPositionLimit[i-1]
    print(f"Joint {name}: [{lower:.3f}, {upper:.3f}] rad")

print(model.gravity.linear)

pin.computeJointJacobians(model, data, q0)
J = pin.getJointJacobian(model, data, model.njoints-1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

M_mat = pin.crba(model, data, q0)
h     = pin.rnea(model, data, q0, v0, a0)
tau_eom = M_mat @ a0 + h

# ── External force to CoM ─────────────────────────────────────────────────
parent_id  = model.getJointId("hip_c_rotation1")
pin.forwardKinematics(model, data, q0)

com_pos    = pin.centerOfMass(model, data, q0)
parent_pos = data.oMi[parent_id].translation
R          = data.oMi[parent_id].rotation

r       = com_pos - parent_pos
F       = np.array([0.0, 0.0, 10.0])
M_cross = np.cross(r, F)

F_local = R.T @ F
M_local = R.T @ M_cross

fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[parent_id] = pin.Force(F_local, M_local)
tau_with_fext = pin.rnea(model, data, q0, v0, a0, fext)

# ── Save model info ───────────────────────────────────────────────────────
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


# =====================================================
# 2. Load CSV data
# =====================================================
csv_files = 'Data_1/resynchronized_data_subject003.csv'
df = pd.read_csv(csv_files)
# =====================================================
# 2. Load CSV data
# =====================================================
csv_files = 'Data_1/resynchronized_data_subject003.csv'
df = pd.read_csv(csv_files)
print("=== CSV loaded ===")
print("Columns in CSV:", df.columns.tolist())

missing = [c for c in all_joints if c not in df.columns]
print(f"Missing from CSV ({len(missing)}): {missing}")

if joint_model:
    df[joint_model] = np.deg2rad(df[joint_model].astype(float))
else:
    print("No joint angle columns found to convert to radians.")

print("after deg2rad:", df[joint_model[0]].head(5).tolist())   # ← check 1

# ── Drop NaN rows before anything else ───────────────────────────────────
df = df.dropna(subset=joint_model).reset_index(drop=True)
df['time'] = df['marker_timestamp'] - df['marker_timestamp'].iloc[0]

print(f"Rows after dropna: {len(df)}")

# ── Build df_filtered ─────────────────────────────────────────────────────
df_filtered = pd.DataFrame()
df_filtered['time'] = df['time'].values

for col in joint_model:
    df_filtered[col] = df[col].values

print("after copy:", df_filtered[joint_model[0]].head(5).tolist())   # ← check 2

# ── Filter coefficients ───────────────────────────────────────────────────
cutoff_freq       = 3
nyquist_freq      = 1 / (2 * df['time'].diff().mean())
normalized_cutoff = cutoff_freq / nyquist_freq
b, a = butter(4, normalized_cutoff, btype='low')

# ── Pass 1: filter ────────────────────────────────────────────────────────
for col in joint_model:
    df_filtered[col] = filtfilt(b, a, df_filtered[col].astype(float))

# ── Pass 2: derivatives ───────────────────────────────────────────────────
for col in joint_model:
    df_filtered['vel_' + col] = df_filtered[col].diff() / df_filtered['time'].diff()
    df_filtered['acc_' + col] = df_filtered['vel_' + col].diff() / df_filtered['time'].diff()
    df_filtered['tau_' + col] = 0.0
    df_filtered[col]          = df_filtered[col].fillna(0)
    df_filtered['vel_' + col] = df_filtered['vel_' + col].replace([np.inf, -np.inf], 0).fillna(0)
    df_filtered['acc_' + col] = df_filtered['acc_' + col].replace([np.inf, -np.inf], 0).fillna(0)

# ── Initialize CoM columns before the loop ───────────────────────────────
df_filtered['com_x'] = 0.0
df_filtered['com_y'] = 0.0
df_filtered['com_z'] = 0.0

# ── Dynamics loop ─────────────────────────────────────────────────────────
for idx, d_i in df_filtered.iterrows():
    q_exp         = d_i[joint_model].to_numpy(dtype=np.float64)
    velocities    = d_i[['vel_' + col for col in joint_model]].to_numpy(dtype=np.float64)
    accelerations = d_i[['acc_' + col for col in joint_model]].to_numpy(dtype=np.float64)

    tau = pin.rnea(model, data, q_exp, velocities, accelerations)
    for col in joint_model:
        df_filtered.at[idx, 'tau_' + col] = tau[joint_model.index(col)]

    com = pin.centerOfMass(model, data, q_exp)
    df_filtered.at[idx, 'com_x'] = com[0]
    df_filtered.at[idx, 'com_y'] = com[1]
    df_filtered.at[idx, 'com_z'] = com[2]

# ── Diagnosis ─────────────────────────────────────────────────────────────
print("=== DIAGNOSIS ===")
print("time unique values  :", df_filtered['time'].nunique())
print("angle unique values :", df_filtered[joint_model[0]].nunique())
print("vel unique values   :", df_filtered['vel_' + joint_model[0]].nunique())
print("com_z unique values :", df_filtered['com_z'].nunique())
print("time head           :", df_filtered['time'].head(5).tolist())
print("angle head          :", df_filtered[joint_model[0]].head(5).tolist())

# =====================================================
# 3. Plots
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
        plt.savefig(filename, dpi=100)
        plt.close(fig)
        print(f"  Saved: {filename}")

    print("=== Saving plots ===")
    save_fig(joint_model,                         "Joint Angles (rad)",          "plot_angles.png")
    save_fig(['vel_' + c for c in joint_model],   "Joint Velocities (rad/s)",    "plot_velocities.png")
    save_fig(['acc_' + c for c in joint_model],   "Joint Accelerations (r/s²)",  "plot_accelerations.png")
    save_fig(['tau_' + c for c in joint_model],   "Joint Torques (Nm)",          "plot_torques.png")
    save_fig(['com_x', 'com_y', 'com_z'],         "CoM Position (m)",            "plot_com.png")
    print("=== All plots saved as PNG ===")