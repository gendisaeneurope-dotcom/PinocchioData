#!/usr/bin/env python3
import numpy as np
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt
import casadi_fun as csf
import ploting_fun as pf
import numpy as _np


# Ensure shapes are (nq, T) / (nv, T)
def ensure_shape(arr, n_dof):
    arr = np.atleast_1d(arr)
    if arr.ndim == 1:
        if arr.size % n_dof == 0:
            arr = arr.reshape(n_dof, -1)
        else:
            arr = arr.reshape(n_dof, -1, order='F')
    elif arr.ndim == 2:
        if arr.shape[0] != n_dof and arr.shape[1] == n_dof:
            arr = arr.T
    else:
        arr = arr.reshape(n_dof, -1)
    return arr



# ---------------------------------------------------------------------------
# 1) Load URDF and build Pinocchio + CasADi dynamics
# ---------------------------------------------------------------------------
root = "world"

# urdf_path = "double_pendulum_simple.urdf"
urdf_path = "double_pendulum_simple.urdf"
# 1) Pinocchio model
model = pin.buildModelFromUrdf(urdf_path)

# 2) Geometry models
collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION)
visual_model    = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL)

data = model.createData()
# We assume fixed-base model here so nq = nv = number of joints
print("nq, nv =", model.nq, model.nv)
assert model.nq == model.nv, "This example assumes a fixed-base model (nq == nv)."
nq = model.nq
nv = model.nv

# CasADi version of the model
cmodel = cpin.Model(model)
cdata = cpin.Data(cmodel)

# CasADi symbols for state variables
q  = ca.SX.sym("q", nq)
dq = ca.SX.sym("dq", nv)
ddq = ca.SX.sym("ddq", nv)

# Torque computed via RNEA
tau = cpin.rnea(cmodel, cdata, q, dq, ddq)

# COM (whole robot)
com = cpin.centerOfMass(cmodel, cdata, q)  # shape (3,)

# Wrap as CasADi Functions
rnea_fun = ca.Function("rnea", [q, dq, ddq], [tau])
com_fun  = ca.Function("com",  [q], [com])

base_id = model.getFrameId("base_link")
N = 120
T = 1
dt = T / N


q0_meas = np.array([0.3683929,  0.52020666])  # initial joint angles
dq0_meas = np.zeros(nv)

q_goal = np.array([-0.43403035, -0.14875698])

# Set the model and data to the current configuration and velocity
pin.forwardKinematics(model, data, q_goal)
pin.updateFramePlacements(model, data)

# Compute the center of mass for q_goal
com_goal = pin.centerOfMass(model, data, q_goal, dq0_meas)

q_init = np.array((np.linspace(1.5708, 2.0399, N), np.linspace(0, -2.2524, N)))  # initial guess
dq_init = np.diff(q_init, axis=1) / dt
ddq_init = np.diff(dq_init, axis=1) / dt 

w = [0, 1, 1, 0.1, 0.1]   # [safety, energy, com_jerk, tau_jerk, joint_jerk]
opti, var = csf.make_pinocchio_model(cmodel, rnea_fun, com_fun, N, w )

csf.instantiate_pinocchio_model(
    var=var,
    opti=opti,
    dt=dt,
    q0=q0_meas,
    dq0=dq0_meas,
    goal_COM=com_goal,
    q_guess=q_init,
    dq_guess=dq_init,
    ddq_guess=ddq_init
)

opti.solver('ipopt')
sol = opti.solve()

q_sol   = sol.value(var['variables']['q'])
dq_sol  = sol.value(var['variables']['dq'])
ddq_sol = sol.value(var['variables']['ddq'])
tau_sol = sol.value(var['functions']['model_tau'])
com_sol = sol.value(var['functions']['COM'])

# Prepare arrays
q_arr = np.array(q_sol)
dq_arr = np.array(dq_sol)
ddq_arr = np.array(ddq_sol)
tau_arr = np.array(tau_sol)
com_arr = np.array(com_sol)

q_arr   = ensure_shape(q_arr, nq)
dq_arr  = ensure_shape(dq_arr, nq)
ddq_arr = ensure_shape(ddq_arr, nq)
tau_arr = ensure_shape(tau_arr, nv)
com_arr = ensure_shape(com_arr, 3)

nvar = csf.numerize_var(var, sol)
if not np.allclose(q_arr, nvar['variables']['q']):
    ValueError('Incoherent q from numerize_var.')


# Time vector (use solver timestep if available)
T_len = q_arr.shape[1]
t = np.arange(T_len) * dt

# Plot results
pf.plot_results(t, q_arr, dq_arr, ddq_arr, tau_arr, com_arr, com_goal)

# Visualization Pinocchio + Meshcat Viewer
# Make sure q has shape (nq, T)
q_sol = np.array(q_sol)
lam_g = np.array(sol.value(opti.lam_g)).flatten()

# Optional: set playback dt (this is NOT the simulation dt)
viewer_dt = 0.03
pf.play_in_meshcat(
    model,
    collision_model,
    visual_model,
    q_sol,
    com_goal,
    com_arr,
    root,
    dt=viewer_dt
)
