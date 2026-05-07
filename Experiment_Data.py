print("=== Script started ===")

import os, signal
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
figure_output = 'Plots'
os.makedirs(figure_output, exist_ok=True)

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

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir)
data = model.createData()

q0 = pin.neutral(model)
v0 = np.zeros(model.nv)
a0 = np.zeros(model.nv)

# Total mass
total_mass = sum([model.inertias[i].mass for i in range(model.njoints)])

# CoM at neutral pose
com0 = pin.centerOfMass(model, data, q0)

# Joint limits
for i in range(1, model.njoints):
    name = model.names[i]
    lower = model.lowerPositionLimit[i-1]
    upper = model.upperPositionLimit[i-1]
    print(f"Joint {name}: [{lower:.3f}, {upper:.3f}] rad")

# Gravity vector
print(model.gravity.linear)

# Jacobian of last joint
pin.computeJointJacobians(model, data, q0)
J = pin.getJointJacobian(model, data, model.njoints-1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

# M(q) — mass matrix
M = pin.crba(model, data, q0)

# h(q, qdot) — gravity and Coriolis at zero velocity

h = pin.rnea(model, data, q0, v0, a0)

# Equation of motion: tau = M(q)*a + h(q,qdot)
tau_eom = M @ a0 + h

# ===========================================================================================
# External force to CoM

# 1. Get parent joint of CoM
parent_id  = model.getJointId("hip_c_rotation1")

# 2. Forward kinematics
pin.forwardKinematics(model, data, q0)

# 3. Where is the CoM and the joint in world space?
com_pos    = pin.centerOfMass(model, data, q0)
parent_pos = data.oMi[parent_id].translation

# 4. Lever arm from joint → CoM
r = com_pos - parent_pos

# 5. Force at CoM (world frame)
F = np.array([0.0, 0.0, 10.0])

# 6. Moment it creates at the parent joint
M = np.cross(r, F)

# 7. Rotate into joint local frame
R = data.oMi[parent_id].rotation
F_local = R.T @ F
M_local = R.T @ M

# 8. Apply and run RNEA
fext = [pin.Force.Zero() for _ in range(model.njoints)]
fext[parent_id] = pin.Force(F_local, M_local)
tau_with_fext = pin.rnea(model, data, q0, v0, a0, fext)

# Save all results to file
with open("model_info.txt", "w") as f:
    # Total mass
    f.write(f"Total mass: {sum([model.inertias[i].mass for i in range(model.njoints)])}\n")

    # CoM at neutral pose
    f.write(f"CoM at neutral: {com0}\n")

    # Joint limits
    f.write("Joint limits:\n")
    for i in range(1, model.njoints):
        name = model.names[i]
        lower = model.lowerPositionLimit[i-1]
        upper = model.upperPositionLimit[i-1]
        f.write(f"  Joint {name}: [{lower:.3f}, {upper:.3f}] rad\n")

    # Gravity vector
    f.write(f"Gravity: {model.gravity.linear}\n")

    # Jacobian of last joint
    f.write(f"Jacobian of last joint:\n{J}\n")

    # M(q)
    f.write(f"Mass matrix M(q):\n{M}\n")

    # h(q, qdot)
    f.write(f"h(q, qdot): {h}\n")

    # Equation of motion
    f.write(f"tau_eom: {tau_eom}\n")

    # External force at CoM
    f.write(f"tau_with_fext: {tau_with_fext}\n")

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

# ── Drop NaN rows before anything else 
df = df.dropna(subset=joint_model).reset_index(drop=True)
df['time'] = df['marker_timestamp'] - df['marker_timestamp'].iloc[0]

print(f"Rows after dropna: {len(df)}")

# ── Build df_filtered 
df_filtered = pd.DataFrame()
df_filtered['time'] = df['time'].values

for col in joint_model:
    df_filtered[col] = df[col].values

print("after copy:", df_filtered[joint_model[0]].head(5).tolist())   # ← check 2

# ── Filter coefficients 
cutoff_freq       = 3
nyquist_freq      = 1 / (2 * df['time'].diff().mean())
normalized_cutoff = cutoff_freq / nyquist_freq
b, a = butter(4, normalized_cutoff, btype='low')

# ── Pass 1: filter 
for col in joint_model:
    df_filtered[col] = filtfilt(b, a, df_filtered[col].astype(float))

# ── Pass 2: derivatives 
for col in joint_model:
    df_filtered['vel_' + col] = df_filtered[col].diff() / df_filtered['time'].diff()
    df_filtered['acc_' + col] = df_filtered['vel_' + col].diff() / df_filtered['time'].diff()
    df_filtered['tau_' + col] = 0.0
    df_filtered[col]          = df_filtered[col].fillna(0)
    df_filtered['vel_' + col] = df_filtered['vel_' + col].replace([np.inf, -np.inf], 0).fillna(0)
    df_filtered['acc_' + col] = df_filtered['acc_' + col].replace([np.inf, -np.inf], 0).fillna(0)

# ── Initialize CoM columns before the loop 
df_filtered['com_x'] = 0.0
df_filtered['com_y'] = 0.0
df_filtered['com_z'] = 0.0

# ── Dynamics loop ─
tau_results      = {col: [] for col in joint_model}
tau_fext_results = {col: [] for col in joint_model}
com_x, com_y, com_z = [], [], []

for idx, d_i in df_filtered.iterrows():
    q_exp         = d_i[joint_model].to_numpy(dtype=np.float64)
    velocities    = d_i[['vel_' + col for col in joint_model]].to_numpy(dtype=np.float64)
    accelerations = d_i[['acc_' + col for col in joint_model]].to_numpy(dtype=np.float64)

    # Torques without external force
    tau = pin.rnea(model, data, q_exp, velocities, accelerations)
    for col in joint_model:
        tau_results[col].append(tau[joint_model.index(col)])

    # ── Time-varying external force ───────────────────────────────────────
    motor_force = float(df.loc[idx, 'motor_force']) if 'motor_force' in df.columns else 0.0

    if motor_force != 0.0:
        pin.forwardKinematics(model, data, q_exp)
        com_pos   = pin.centerOfMass(model, data, q_exp)
        joint_pos = data.oMi[parent_id].translation
        joint_rot = data.oMi[parent_id].rotation
        r         = com_pos - joint_pos
        F         = np.array([0.0, motor_force, 0.0])   # Y-axis (lateral perturbation)
        M_cross   = np.cross(r, F)
        fext      = [pin.Force.Zero() for _ in range(model.njoints)]
        fext[parent_id] = pin.Force(joint_rot.T @ F, joint_rot.T @ M_cross)
        tau_fext  = pin.rnea(model, data, q_exp, velocities, accelerations, fext)
        for col in joint_model:
            tau_fext_results[col].append(tau_fext[joint_model.index(col)])
    else:
        for col in joint_model:
            tau_fext_results[col].append(0.0)

    # CoM
    com = pin.centerOfMass(model, data, q_exp)
    com_x.append(com[0])
    com_y.append(com[1])
    com_z.append(com[2])

# ── Bulk assign after loop ────────────────────────────────────────────────
for col in joint_model:
    df_filtered['tau_' + col]      = tau_results[col]
    df_filtered['tau_fext_' + col] = tau_fext_results[col]
df_filtered['com_x'] = com_x  # left-right
df_filtered['com_y'] = com_y  # forward-backward
df_filtered['com_z'] = com_z  # vertical (height)

# Diagnosis 
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
        plt.savefig(os.path.join(figure_output, filename), dpi=100)
        plt.close(fig)
        print(f"  Saved: {filename}")

    print("=== Saving plots ===")
    save_fig(joint_model,                         "Joint Angles (rad)",          "plot_angles.png")
    save_fig(['vel_' + c for c in joint_model],   "Joint Velocities (rad/s)",    "plot_velocities.png")
    save_fig(['acc_' + c for c in joint_model],   "Joint Accelerations (r/s²)",  "plot_accelerations.png")
    save_fig(['tau_' + c for c in joint_model],   "Joint Torques (Nm)",          "plot_torques.png")
    save_fig(['com_x', 'com_y', 'com_z'],         "CoM Position (m)",            "plot_CoM.png")
    print("=== All plots saved as PNG ===")

# =====================================================
# 4. Trajectory loop + plots
# =====================================================

print("-" * 60)
print("CSV LOADED — starting trajectory analysis")

# Load CSV
csv_path = 'Data_1/resynchronized_data_subject003.csv'  # ← replace with your filename
df = pd.read_csv(csv_path, low_memory=False)
df["t_sync"] = pd.to_numeric(df["t_sync"], errors="coerce")
df = df.dropna(subset=["t_sync"]).reset_index(drop=True)
df["t_sync"] = df["t_sync"] - df["t_sync"].iloc[0]

# Compute angles
deg2rad = np.pi / 180.0
df['Single_leg_hip_abd']    = 0.5 * (df['hip_adduction_r'] - df['hip_adduction_l']) * deg2rad
df['Single_leg_hip_flex']   = 0.5 * (df['hip_flexion_r']   + df['hip_flexion_l'])   * deg2rad
df['Single_leg_ankle_flex'] = df['subtalar_sagittal_tilt_rad']
df['Single_leg_ankle_abd']  = df['subtalar_frontal_tilt_rad']

time = df['t_sync'].values
print(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Time range: {time[0]:.3f} → {time[-1]:.3f} s")
print("-" * 60)

print(df.columns[:10].tolist())  # first 10 columns
print(df.columns[-10:].tolist())  # last 10 columns
print(df['marker_timestamp'].unique()[:10])

# Find where t_sync resets (new trial starts)
t_sync = df["t_sync"].values
restarts = np.where(np.diff(t_sync) < 0)[0] + 1
print(f"Number of trials: {len(restarts) + 1}")
print(f"Trial start indices: {restarts}")

print(df["t_sync"].iloc[:5].tolist())   # start of first trial
if len(restarts) > 0:
    print(df["t_sync"].iloc[restarts[0]-2 : restarts[0]+3].tolist())  # around first restart

# Full time range
print(f"Full time range: {t_sync[0]:.2f} → {t_sync[-1]:.2f} s")
print(f"Total duration: {t_sync[-1] - t_sync[0]:.2f} s")
print(f"Total rows: {len(df)}")
print(f"Sampling rate: {1 / np.mean(np.diff(t_sync)):.1f} Hz")

# =====================================================
# 5. Visualization setup with MeshCat
# =====================================================

viz = MeshcatVisualizer(model, collision_model, visual_model)

viz.initViewer(zmq_url="tcp://127.0.0.1:6001", open=False)

time_module.sleep(2)
viz.loadViewerModel(rootNodeName=root)
print("MeshCat open at: http://127.0.0.1:7001/static/")

q0 = pin.neutral(model)
viz.display(q0)                                # show skeleton at neutral pose first

rate = 1 / 60                                  # ~60 fps animation

# Set CoM sphere geometry once, before the loop
viz.viewer['com'].set_object(
    g.Sphere(0.08),
    g.MeshLambertMaterial(color=0xff0000))

try:
    for i in range(len(df_filtered)):
        q = df_filtered[joint_model].iloc[i].values.astype(float)
        viz.display(q)

        com = df_filtered[['com_x', 'com_y', 'com_z']].iloc[i].values.astype(float)
        viz.viewer['com'].set_transform(
            tf.translation_matrix(com))

        time_module.sleep(rate)
finally:
    proc = viz.viewer.window.server_proc
    proc.terminate()          # polite shutdown first
    try:
        proc.wait(timeout=3)  # give it 3 seconds
    except Exception:
        os.kill(proc.pid, signal.SIGTERM)  # force if still alive
    print("MeshCat stopped. Script finished.")

#plt.show()