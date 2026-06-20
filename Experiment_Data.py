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

# CoM columns before the loop 
df_filtered['com_x'] = 0.0
df_filtered['com_y'] = 0.0
df_filtered['com_z'] = 0.0
df_filtered['com_error_x'] = 0.0
df_filtered['com_error_y'] = 0.0
df_filtered['com_error_z'] = 0.0

# ── Dynamics loop ─
tau_results      = {col: [] for col in joint_model}
tau_fext_results = {col: [] for col in joint_model}
com_x, com_y, com_z = [], [], []
com_error_x, com_error_y, com_error_z = [], [], []

for idx, d_i in df_filtered.iterrows():
    q_exp         = d_i[joint_model].to_numpy(dtype=np.float64)
    velocities    = d_i[['vel_' + col for col in joint_model]].to_numpy(dtype=np.float64)
    accelerations = d_i[['acc_' + col for col in joint_model]].to_numpy(dtype=np.float64)

    # 1. Torques without external force (RNEA)
    tau = pin.rnea(model, data, q_exp, velocities, accelerations)
    for col in joint_model:
        tau_results[col].append(tau[joint_model.index(col)])

    # 2. CoM position
    com = pin.centerOfMass(model, data, q_exp)
    com_x.append(com[0])
    com_y.append(com[1])
    com_z.append(com[2])

    # 3. CoM error (desired = no lateral drift, same height)
    com_desired = np.array([0.0, 0.0, com[2]])   # stay centered, keep height
    com_error   = com_desired - com
    com_error_x.append(com_error[0])
    com_error_y.append(com_error[1])
    com_error_z.append(com_error[2])

    # 4. Torques WITH external perturbation force ───────────────────────────────────────
    motor_force = float(df.loc[idx, 'motor_force']) if 'motor_force' in df.columns else 0.0

    if motor_force != 0.0:
        pin.forwardKinematics(model, data, q_exp)
        com_pos    = pin.centerOfMass(model, data, q_exp)
        joint_pos  = data.oMi[parent_id].translation
        joint_rot  = data.oMi[parent_id].rotation
        r          = com_pos - joint_pos
        F          = np.array([0.0, motor_force, 0.0])   # Y-axis (lateral perturbation)
        M          = np.cross(r, F)
        fext       = [pin.Force.Zero() for _ in range(model.njoints)]
        fext[parent_id] = pin.Force(joint_rot.T @ F, joint_rot.T @ M)
        tau_fext = pin.rnea(model, data, q_exp, velocities, accelerations, fext) 
        
        for col in joint_model:
            tau_fext_results[col].append(tau_fext[joint_model.index(col)])
    else:
        for col in joint_model:
            tau_fext_results[col].append(0.0)


# ── Bulk assign after loop ────────────────────────────────────────────────
for col in joint_model:
    df_filtered['tau_' + col]      = tau_results[col]
    df_filtered['tau_fext_' + col] = tau_fext_results[col]
df_filtered['com_x'] = com_x  # left-right
df_filtered['com_y'] = com_y  # forward-backward
df_filtered['com_z'] = com_z  # vertical (height)
df_filtered['com_error_x'] = com_error_x  # left-right
df_filtered['com_error_y'] = com_error_y  # forward-backward
df_filtered['com_error_z'] = com_error_z  # vertical (height)

# ── Copy block/trial info from original df ───────────────────────────────
df_filtered['block_id'] = df['block_idx'].round(-3).values   #rounds to the nearest 1000
df_filtered['motor_force'] = df['motor_force'].values

df_filtered.to_csv('Data_1/Subject3_processed.csv', index=False)
print("Saved → Data_1/Subject3_processed.csv")

# ── feature matrix ───────────────────────────────────────────────
df_out = pd.DataFrame({
    'time':         df_filtered['time'].values,
    'block_id':     df_filtered['block_id'].values,
    'com_x':        com_x,
    'com_y':        com_y,
    'com_z':        com_z,
    'com_error_x':  com_error_x,
    'com_error_y':  com_error_y,
    'com_error_z':  com_error_z,
    'motor_force':  df_filtered['motor_force'].values
})

for col in joint_model:
    df_out['tau_' + col]      = tau_results[col]
    df_out['tau_fext_' + col] = tau_fext_results[col]

df_out.to_csv('Data_1/Subject3_features.csv', index=False)
print("Saved → Data_1/Subject3_features.csv")

# ── PD Controller ─────────────────────────────────────────────────────────
Kp = 100.0
Kd = 10.0

df_out['com_error_y_dot'] = df_out['com_error_y'].diff() / df_out['time'].diff()
df_out['com_error_y_dot'] = df_out['com_error_y_dot'].fillna(0)
df_out['tau_PD']          = Kp * df_out['com_error_y'] + Kd * df_out['com_error_y_dot']

df_out.to_csv('Data_1/Subject3_features.csv', index=False)
print("PD torques added → Subject3_features.csv")

# ── Scale motor_force from mN to N ───────────────────────────────────────
df['motor_force']           = df['motor_force'] / 1000.0
df_filtered['motor_force']  = df_filtered['motor_force'] / 1000.0
df_out['motor_force']       = df_out['motor_force'] / 1000.0

# ── Jᵀ mapping ───────────────────────────────────────────────────────────
tau_Jt_results = {col: [] for col in joint_model}

for idx, d_i in df_filtered.iterrows():
    q_exp       = d_i[joint_model].to_numpy(dtype=np.float64)
    motor_force = float(df.loc[idx, 'motor_force']) if 'motor_force' in df.columns else 0.0

    if motor_force != 0.0:
        pin.forwardKinematics(model, data, q_exp)
        pin.computeJointJacobians(model, data, q_exp)
        J_CoM        = pin.jacobianCenterOfMass(model, data, q_exp)
        F            = np.array([0.0, motor_force, 0.0])
        tau_de_force = J_CoM.T @ F
        for col in joint_model:
            tau_Jt_results[col].append(tau_de_force[joint_model.index(col)])
    else:
        for col in joint_model:
            tau_Jt_results[col].append(0.0)

for col in joint_model:
    df_out['tau_Jt_' + col] = tau_Jt_results[col]

df_out.to_csv('Data_1/Subject3_features.csv', index=False)
print("Jᵀ torques added → Subject3_features.csv")



