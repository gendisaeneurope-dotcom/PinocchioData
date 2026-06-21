print("=== Script started ===")

import os
print("os OK")
import sys
print("sys OK")
from pathlib import Path
print("pathlib OK")
from matplotlib.pylab import rand
print("matplotlib OK")
import numpy as np
print("numpy OK")
import matplotlib.pyplot as plt
print("pyplot OK")
import pandas as pd
print("pandas OK")
import pinocchio as pin
print("pinocchio OK")
from pinocchio.visualize import MeshcatVisualizer
print("MeshcatVisualizer OK")
import time
print("time OK")
from scipy.signal import butter, filtfilt
print("scipy OK")

print("=== Imports done ===")

root = "world"
plot = True

all_joints = ['L_ankle_flex','L_ankle_abd', 'L_knee_flex', 'L_hip_abd', 'L_hip_flex',
              'R_ankle_flex','R_ankle_abd','R_knee_flex', 'R_hip_abd', 'R_hip_flex',
              'Single_leg_ankle_abd', 'Single_leg_hip_abd']

joint_model = [
            'Single_leg_ankle_abd',    
            'Single_leg_hip_abd',       
            ]


# =====================================================
# 1. Build Pinocchio model
# =====================================================
mesh_dir = os.path.dirname(os.path.abspath(__file__))
urdf_filename = "human_subject3.urdf"
urdf_model_path = os.path.join(mesh_dir, urdf_filename)

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir)
data = model.createData()

print("=== Model loaded ===")

# Basic model info
print("model.nq:", model.nq)
print("model.nv:", model.nv)
print("Joint names:", [model.names[i] for i in range(model.njoints)])

# Total mass
total_mass = sum([model.inertias[i].mass for i in range(model.njoints)])
print("Total mass:", total_mass, "kg")

# Gravity vector
print("Gravity vector:", model.gravity.linear)

# Joint limits
for i in range(1, model.njoints):
    name = model.names[i]
    lower = model.lowerPositionLimit[i-1]
    upper = model.upperPositionLimit[i-1]
    print(f"Joint {name}: [{lower:.3f}, {upper:.3f}] rad")

# CoM at neutral pose
q0 = pin.neutral(model)
com0 = pin.centerOfMass(model, data, q0)
print("CoM at neutral pose:", com0)

# Jacobian of last joint at neutral pose
pin.computeJointJacobians(model, data, q0)
J = pin.getJointJacobian(model, data, model.njoints-1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
print("Jacobian of last joint:\n", J)

# =====================================================
# Equation of Motion checks
# =====================================================
v0 = np.zeros(model.nv)
a0 = np.zeros(model.nv)

# Mass matrix M(q)
M = pin.crba(model, data, q0)
print("Mass matrix M(q):\n", M)

# h(q, qdot) — gravity + Coriolis at zero velocity
h = pin.rnea(model, data, q0, v0, a0)
print("h(q, qdot):", h)

# Equation of motion: tau = M(q)*a + h(q, qdot)
tau_eom = M @ a0 + h
print("Torque from equation of motion:", tau_eom)

# External force at last joint (Fz = 10N)
fext = pin.StdVec_Force()
for i in range(model.njoints):
    fext.append(pin.Force.Zero())
fext[model.njoints - 1] = pin.Force(np.array([0, 0, 10, 0, 0, 0]))
tau_with_fext = pin.rnea(model, data, q0, v0, a0, fext)
print("Torque with external force:", tau_with_fext)

# =====================================================
# 2. Load CSV data
# =====================================================
csv_files = 'Data/trial_GO_TO_RIGHT_CIRCLE_AFTER_TRIAL_91728_92027.csv'
df = pd.read_csv(csv_files)
print("=== CSV loaded ===")
print("Columns in CSV:", df.columns.tolist())

if joint_model:
    df[joint_model] = np.deg2rad(df[joint_model].astype(float))
else:
    print("No joint angle columns found to convert to radians.")

df['time'] = df['marker_timestamp'] - df.head(1)['marker_timestamp'].values[0]  

df_filtered = pd.DataFrame()
df_filtered['time'] = df['time']
for col in joint_model:
    df_filtered[col] = df[col]

cutoff_freq = 3
nyquist_freq = 1 / (2 * df['time'].diff().mean())
normalized_cutoff = cutoff_freq / nyquist_freq
b, a = butter(4, normalized_cutoff, btype='low')

for col in joint_model:
    df_filtered[col] = filtfilt(b, a, df_filtered[col].astype(float))
    df_filtered['vel_' + col] = df_filtered[col].diff() / df_filtered['time'].diff()
    df_filtered['acc_' + col] = df_filtered['vel_' + col].diff() / df_filtered['time'].diff()
    df_filtered['tau_' + col] = 0.0
    df_filtered[col] = df_filtered[col].fillna(0)
    df_filtered['vel_' + col] = df_filtered['vel_' + col].replace([np.inf, -np.inf], 0).fillna(0)
    df_filtered['acc_' + col] = df_filtered['acc_' + col].replace([np.inf, -np.inf], 0).fillna(0)

for idx, d_i in df_filtered.iterrows():
    q_exp = np.asarray(np.array(d_i[joint_model]), dtype=np.float64)
    velocities = np.asarray(np.array(d_i[['vel_' + col for col in joint_model]]), dtype=np.float64)
    accelerations = np.asarray(np.array(d_i[['acc_' + col for col in joint_model]]), dtype=np.float64)
    tau = pin.rnea(model, data, q_exp, velocities, accelerations)
    for col in joint_model:
        df_filtered.at[idx, 'tau_' + col] = tau[joint_model.index(col)]

if plot:
    if all(angle in df_filtered.columns for angle in joint_model):
        fig2, ax2 = plt.subplots()
        df_filtered[joint_model].plot(ax=ax2, title="Joint Angles")
        fig3, ax3 = plt.subplots()
        df_filtered[['vel_' + col for col in joint_model]].plot(ax=ax3, title="Joint Velocities")
        fig4, ax4 = plt.subplots()
        df_filtered[['acc_' + col for col in joint_model]].plot(ax=ax4, title="Joint Accelerations")
        fig5, ax5 = plt.subplots()
        df_filtered[['tau_' + col for col in joint_model]].plot(ax=ax5, title="Joint Torques")
        
        plt.figure()
        plt.plot(df_filtered['time'], df[all_joints])
        plt.legend(all_joints)
        # plt.show(block=False)
        


# =====================================================
# 3. Visualization setup with MeshCat
# =====================================================
print("=== Press Enter to start MeshCat visualization ===")
input() 

import meshcat.geometry as g
import meshcat.transformations as tf

viz = MeshcatVisualizer(model, collision_model, visual_model)

try:
    viz.initViewer(open=True)  # opens browser automatically
except ImportError as err:
    print("Error while initializing the viewer. Install meshcat with: pip install meshcat")
    print(err)
    sys.exit(0)

viz.loadViewerModel()
print("Open MeshCat at: http://127.0.0.1:7000/static/")

q0 = pin.neutral(model)
viz.display(q0)

rate = 1/60

for i in range(len(df_filtered)):
    q = df_filtered[joint_model].iloc[i].values.astype(float)
    viz.display(q)

    com = pin.centerOfMass(model, data, q)
    viz.viewer['com'].set_object(g.Sphere(0.08), 
                                  g.MeshLambertMaterial(color=0xff0000))
    viz.viewer['com'].set_transform(tf.translation_matrix([com[0], com[1], com[2]]))

    time.sleep(rate)
    
plt.show()
