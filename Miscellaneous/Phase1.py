# Phase 1 #
# Objectives: 
# 1. Implement task-space PD controller, compute CoM error, map force to torques via Jᵀ, test sinusoidal lateral motion
# 2. Apply forward force at CoM, simulate disturbances, observe controller compensation (hip/ankle torques)

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

# =====================================================
# 1. Load model
# =====================================================
mesh_dir        = os.path.dirname(os.path.abspath(__file__))
urdf_filename   = "subject3_single_leg_4dof.urdf"
urdf_model_path = os.path.join(mesh_dir, urdf_filename)

model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)
data = model.createData()

joint_model = [
    'Single_leg_ankle_abd',
    'Single_leg_hip_abd',
    'Single_leg_hip_flex',
    'Single_leg_ankle_flex'
]

# =====================================================
# 2. Simulation constants
# =====================================================
dt     = 0.01       # timestep (s) — 100 Hz
N      = 1000       # max steps = 10 seconds
Kp     = 100.0      # proportional gain
Kd     = 10.0       # derivative gain
thresh = 1e-3       # early stop threshold (m)

# Sinusoidal CoM reference
A = 0.05            # amplitude (m)
f = 0.5             # frequency (Hz)

# External lateral force — applied after halfway point
F_external  = np.array([0.0, 10.0, 0.0])   # 10N in Y (lateral)
force_start = N // 2                         # turn on at step 500

# Parent joint for force application
parent_id = model.getJointId("hip_c_rotation1")

# =====================================================
# 3. Initial state
# =====================================================
q  = pin.neutral(model).copy()
dq = np.zeros(model.nv)

# =====================================================
# 4. Simulation loop
# Task-space PD controller following sinusoidal CoM
# goal. After halfway, a lateral force is applied.
# Observe how hip/ankle torques respond.
# =====================================================
com_goal_history   = []
com_actual_history = []
error_history      = []

# Per-joint torque histories
tau_hip_abd_history   = []
tau_ankle_flex_history = []
tau_fext_hip_abd_history   = []
tau_fext_ankle_flex_history = []

print(f"{'Step':>6} | {'Goal Y':>10} | {'Actual Y':>10} | {'Error Y':>10} | {'Force':>8}")
print("-" * 60)

for step in range(N):
    t = step * dt

    # --- Sinusoidal CoM goal (Y axis — lateral) ---
    com_goal_y = A * np.sin(2 * np.pi * f * t)

    # --- Current CoM ---
    pin.forwardKinematics(model, data, q)
    com          = pin.centerOfMass(model, data, q)
    com_actual_y = com[1]

    # --- CoM error ---
    error_y     = com_goal_y - com_actual_y
    error_y_dot = -dq.mean()    # velocity proxy for derivative term

    # --- Task-space PD torque ---
    tau_PD = Kp * error_y + Kd * error_y_dot

    # --- Map PD scalar to joint torques via Jᵀ ---
    pin.computeJointJacobians(model, data, q)
    J_com        = pin.jacobianCenterOfMass(model, data, q)
    F_pd         = np.array([0.0, tau_PD, 0.0])   # PD force in Y direction
    tau          = J_com.T @ F_pd                  # map to joint torques via Jᵀ

    # --- Record PD torques per joint ---
    tau_hip_abd_history.append(tau[joint_model.index('Single_leg_hip_abd')])
    tau_ankle_flex_history.append(tau[joint_model.index('Single_leg_ankle_flex')])

    # --- Apply external lateral force after halfway ---
    USE_FORCE  = (step >= force_start)
    tau_fext_J = np.zeros(model.nv)
    if USE_FORCE:
        com_pos    = pin.centerOfMass(model, data, q)
        joint_pos  = data.oMi[parent_id].translation
        joint_rot  = data.oMi[parent_id].rotation
        r          = com_pos - joint_pos
        M_cross    = np.cross(r, F_external)
        fext       = [pin.Force.Zero() for _ in range(model.njoints)]
        fext[parent_id] = pin.Force(joint_rot.T @ F_external, joint_rot.T @ M_cross)

        # Jᵀ mapping of external force
        tau_fext_J = J_com.T @ F_external

    tau_fext_hip_abd_history.append(tau_fext_J[joint_model.index('Single_leg_hip_abd')])
    tau_fext_ankle_flex_history.append(tau_fext_J[joint_model.index('Single_leg_ankle_flex')])

    # --- Total torque = PD compensation + disturbance ---
    tau_total = tau + tau_fext_J

    # --- Forward dynamics: ddq = M⁻¹(tau_total - h) ---
    M_mat = pin.crba(model, data, q)
    h     = pin.rnea(model, data, q, dq, np.zeros(model.nv))
    ddq   = np.linalg.solve(M_mat, tau_total - h)

    # --- Euler integration ---
    dq += ddq * dt
    q  += dq  * dt
    q   = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

    # --- Record ---
    com_goal_history.append(com_goal_y)
    com_actual_history.append(com_actual_y)
    error_history.append(error_y)

    if step % 100 == 0:
        print(f"{step:>6} | {com_goal_y:>10.5f} | {com_actual_y:>10.5f} | {error_y:>10.5f} | {'ON' if USE_FORCE else 'OFF':>8}")

    # --- Early stop ---
    if abs(error_y) < thresh and step > 50:
        print(f"\n✅ Converged at step {step}")
        break

# =====================================================
# 5. Plots
# =====================================================
time_axis  = np.arange(len(com_goal_history)) * dt
force_time = force_start * dt

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle('Phase 1 — PD Controller: Sinusoidal Tracking + Lateral Disturbance', fontsize=13)

# Plot 1: CoM tracking
axes[0].plot(time_axis, com_goal_history,   label='Desired CoM Y (sinusoid)', linestyle='--', color='blue')
axes[0].plot(time_axis, com_actual_history, label='Actual CoM Y (PD)',        color='orange')
axes[0].axvline(x=force_time, color='red', linestyle=':', linewidth=1.2, label='Force ON')
axes[0].set_ylabel('CoM Y (m)')
axes[0].legend()
axes[0].set_title('CoM Tracking')

# Plot 2: tracking error
axes[1].plot(time_axis, error_history, label='Tracking error Y', color='red')
axes[1].axhline(0, color='k', linestyle='--', linewidth=0.8)
axes[1].axvline(x=force_time, color='red', linestyle=':', linewidth=1.2)
axes[1].set_ylabel('Error (m)')
axes[1].legend()
axes[1].set_title('Tracking Error')

# Plot 3: hip abd + ankle flex torques (PD vs force compensation)
axes[2].plot(time_axis, tau_hip_abd_history,        label='Hip Abd — tau_PD',         color='blue',   alpha=0.8)
axes[2].plot(time_axis, tau_ankle_flex_history,     label='Ankle Flex — tau_PD',      color='green',  alpha=0.8)
axes[2].plot(time_axis, tau_fext_hip_abd_history,   label='Hip Abd — tau_fext (Jᵀ)',  color='navy',   linestyle='--')
axes[2].plot(time_axis, tau_fext_ankle_flex_history,label='Ankle Flex — tau_fext (Jᵀ)',color='darkgreen', linestyle='--')
axes[2].axvline(x=force_time, color='red', linestyle=':', linewidth=1.2, label='Force ON')
axes[2].axhline(0, color='k', linestyle='--', linewidth=0.8)
axes[2].set_ylabel('Torque (N·m)')
axes[2].set_xlabel('Time (s)')
axes[2].legend()
axes[2].set_title('Hip Abd + Ankle Flex: PD Torque vs Force Compensation')

plt.tight_layout()
plt.savefig('phase1_pd_sinusoidal.png', dpi=120)
print("Saved → phase1_pd_sinusoidal.png")
