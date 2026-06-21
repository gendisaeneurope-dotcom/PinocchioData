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

# External force at last joint (Fz = 10N)
pin.forwardKinematics(model, data, q0)       
fext = [pin.Force.Zero() for _ in range(model.njoints)]  

joint_id = model.njoints - 1                    # last joint for now

F_local = np.array([0.0, 0.0, 10.0])           # N in Z, joint local frame
M_local = np.array([0.0, 0.0, 0.0])            # Nm

fext[joint_id] = pin.Force(F_local, M_local)

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

    # External force at last joint
    f.write(f"tau_with_fext: {tau_with_fext}\n")

model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)
data = model.createData()

# ── Inspect your model — run this ONCE to know your joint names ──
print("=== JOINTS ===")
for i in range(model.njoints):
    print(f"  [{i}] {model.names[i]}")

print("\n=== FRAMES ===")
for i, frame in enumerate(model.frames):
    print(f"  [{i}] {frame.name}  →  parent joint: {model.names[frame.parentJoint]}")
