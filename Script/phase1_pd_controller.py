# =====================================================
# Phase 1 — Task-Space PD Controller
# Objectives:
# 1. Define static/sinusoidal goal CoM (no real data)
# 2. PD loop INSIDE dynamics loop
# 3. Control both X and Y CoM axes (gain = vector)
# 4. External force applied AFTER controller step, so controller can compensate for disturbance
# =====================================================
print("=== Phase 1 started ===")
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
print("=== Imports done ===")


# =====================================================
# 1. Load model
# =====================================================
mesh_dir        = os.path.dirname(os.path.abspath(__file__))
urdf_filename   = "subject3_single_leg_4dof.urdf"
urdf_model_path = os.path.join(mesh_dir, urdf_filename)

model, _, _ = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)
data        = model.createData()

JOINT_MODEL = [
    'Single_leg_ankle_abd',
    'Single_leg_hip_abd',
    'Single_leg_hip_flex',
    'Single_leg_ankle_flex',
]

parent_id = model.getJointId("hip_c_rotation1")
print(f"Model loaded — {model.njoints} joints, nv={model.nv}")


# =====================================================
# 2. Simulation constants
# =====================================================
DT  = 0.01    # timestep (s) — 100 Hz
N   = 1000    # max steps  = 10 s

# Per-axis PD gains — derived from model physics
# Second-order critical damping: KP = m*omega_n^2, KD = 2*m*omega_n
# omega_n chosen to match human balance recovery speed
# X control disabled — model has no natural restoring force in X,
# sagittal axis is hard-clamped near neutral instead
total_mass = sum(model.inertias[i].mass for i in range(model.njoints))
print(f"Total model mass: {total_mass:.2f} kg")

omega_n_Y = 2 * np.pi * 0.5        # 0.5 Hz — lateral balance response

KP_Y = 2000.0 
KD_Y = 400.0 # reduced from critical damping to allow some oscillation — more human-like, and better test of controller response to impulse
KP_X = 60.0   # X PD disabled — controlled via hard clamp in integration step
KD_X = 10.0

print(f"Derived gains — KP_Y={KP_Y:.2f}  KD_Y={KD_Y:.2f}")
print(f"                KP_X=disabled (hard clamp)")

# Joint saturation limits
TAU_MAX = 200.0   # Nm — scaled for 67 kg model
DQ_MAX  = 3.0     # rad/s — max joint velocity

COM_ERROR_TOLERANCE = 1e-6   # early-stop: both X and Y errors below 1 mm

# Sinusoidal CoM goal — lateral Y oscillation, hold X at 0
com_rest_X = 0.00126 # model's natural CoM is slightly forward of hip joint, so add small offset to prevent constant forward error and large PD response
com_rest_Y = 0.02893 
SIN_AMP    = 0.05   # amplitude (m) — 5 cm lateral sway
SIN_FREQ   = 0.5    # frequency (Hz)
com_goal_x = com_rest_X   # hold X at neutral throughout


# External lateral force — ONE timestep impulse at midpoint
# Applied AFTER controller so controller can compensate next step
F_EXTERNAL  = np.array([0.0, -20.0, 0.0])  # 50N in Y (lateral) — scaled for 67 kg model to create noticeable but recoverable disturbance
IMPULSE_DURATION = 0.05                    # 50 ms impulse duration → convert to equivalent force magnitude 
FORCE_STEP  = N // 2                       # midpoint of sinusoid


# =====================================================
# 3. Initial state
# =====================================================
q  = pin.neutral(model).copy()
dq = np.zeros(model.nv)

# check initial CoM position
pin.forwardKinematics(model, data, q)
com_initial = pin.centerOfMass(model, data, q)
print(f"Initial CoM: {com_initial}")

pin.computeJointJacobians(model, data, q)
J_com = pin.jacobianCenterOfMass(model, data, q)
print(f"J_com shape: {J_com.shape}")
print(f"J_com norm: {np.linalg.norm(J_com):.4f}")
print(f"Max Jt@F magnitude: {np.max(np.abs(J_com.T @ np.array([0,1,0]))):.4f}")

# Initialise previous errors for derivative term
previous_error_x = com_rest_X - float(com_initial[0])
previous_error_y = com_rest_Y - float(com_initial[1])


# =====================================================
# 4. Simulation loop
# PD inside loop → force after controller → integrate
# =====================================================
history = {
    'com_goal_x':   [],
    'com_goal_y':   [],
    'com_actual_x': [],
    'com_actual_y': [],
    'error_x':      [],
    'error_y':      [],
    'tau_pd':       {c: [] for c in JOINT_MODEL},
    'tau_fext':     {c: [] for c in JOINT_MODEL},
    'force_active': [],
}

print(f"\n{'Step':>6} | {'GoalY':>8} | {'ActY':>8} | {'ErrY':>8} | {'Force':>6}")
print("-" * 50)

for step in range(N):
    t = step * DT

    # ── Step A: sinusoidal CoM goal ───────────────────────────
    com_goal_y = com_rest_Y + SIN_AMP * np.sin(2 * np.pi * SIN_FREQ * t)

    # ── Step B: current CoM from forward kinematics ──────────
    pin.forwardKinematics(model, data, q)
    com          = pin.centerOfMass(model, data, q)
    com_actual_x = float(com[0])
    com_actual_y = float(com[1])

    # ── Step C: PD controller (X and Y independently) ────────
    error_x = com_goal_x - com_actual_x
    error_y = com_goal_y  - com_actual_y

    # Derivative: finite difference of error
    derivative_x = (error_x - previous_error_x) / DT
    derivative_y = (error_y - previous_error_y) / DT

    # Gravity compensation — cancel gravity torques so PD only handles tracking
    tau_gravity = pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))

    F_pd_x = KP_X * error_x + KD_X * derivative_x
    F_pd_y = KP_Y * error_y + KD_Y * derivative_y

    F_pd = np.array([F_pd_x, F_pd_y, 0.0])

    # Map PD force to joint torques via Jᵀ
    pin.computeJointJacobians(model, data, q)
    J_com  = pin.jacobianCenterOfMass(model, data, q)
    tau_pd = J_com.T @ F_pd

    for c in JOINT_MODEL:
        history['tau_pd'][c].append(tau_pd[JOINT_MODEL.index(c)])

    # Save errors for next derivative
    previous_error_x = error_x
    previous_error_y = error_y

    # ── Step D: external force AFTER controller ───────────────
    # Controller already computed its response to current state.
    # Force is applied as an additional torque disturbance.
    # On next step the controller will see the displaced CoM and generate compensating torques
    USE_FORCE  = (step >= FORCE_STEP)   # one-timestep impulse
    tau_fext   = np.zeros(model.nv)

    if USE_FORCE:
        com_pos   = pin.centerOfMass(model, data, q)
        jnt_pos   = data.oMi[parent_id].translation
        jnt_rot   = data.oMi[parent_id].rotation
        r         = com_pos - jnt_pos
        fext_list = [pin.Force.Zero() for _ in range(model.njoints)]
        fext_list[parent_id] = pin.Force(
            jnt_rot.T @ F_EXTERNAL,
            jnt_rot.T @ np.cross(r, F_EXTERNAL)
        )
        tau_fext = J_com.T @ F_EXTERNAL

    for c in JOINT_MODEL:
        history['tau_fext'][c].append(tau_fext[JOINT_MODEL.index(c)])
    history['force_active'].append(1 if USE_FORCE else 0)

    # ── Step E: total torque = PD + disturbance ───────────────
    tau_total = tau_pd + tau_fext + tau_gravity  # add gravity compensation back in for full torque
    tau_total = np.clip(tau_total, -TAU_MAX, TAU_MAX)
    
    if step == 0:
        print(f"Step 0 tau_total: {tau_total}")

    # ── Step F: forward dynamics M⁻¹(tau - h) ────────────────
    M_mat = pin.crba(model, data, q)
    h     = pin.rnea(model, data, q, dq, np.zeros(model.nv))

    # impulse force also directly perturbs acceleration
    ddq = np.linalg.solve(M_mat, tau_total - h)
    if USE_FORCE:
        ddq += np.linalg.solve(M_mat, J_com.T @ F_EXTERNAL)

    # ── Step G: Euler integration ─────────────────────────────
    dq += ddq * DT
    dq  = np.clip(dq, -DQ_MAX, DQ_MAX)
    q  += dq  * DT
    q   = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
    q[0] = np.clip(q[0], -0.3, 0.3)   # hard-clamp sagittal — replaces X PD

    # ── Record ────────────────────────────────────────────────
    history['com_goal_x'].append(com_goal_x)
    history['com_goal_y'].append(com_goal_y)
    history['com_actual_x'].append(com_actual_x)
    history['com_actual_y'].append(com_actual_y)
    history['error_x'].append(error_x)
    history['error_y'].append(error_y)

    if step % 100 == 0:
        print(f"{step:>6} | {com_goal_y:>8.4f} | {com_actual_y:>8.4f} | "
              f"{error_y:>8.4f} | {'IMPULSE' if USE_FORCE else 'off':>6}")

    # ── Early stop ────────────────────────────────────────────
    if abs(error_y) < COM_ERROR_TOLERANCE and abs(error_x) < COM_ERROR_TOLERANCE and step > 50:
        print(f"\nConverged at step {step} ({t:.2f}s)")
        break

n_steps = len(history['com_goal_y'])
time_axis  = np.arange(n_steps) * DT
force_time = FORCE_STEP * DT
print(f"\nSimulation finished — {n_steps} steps ({n_steps * DT:.1f}s)")

# ── Impulse debug ─────────────────────────────────────────────────────────
print(f"Impulse active frames: {sum(history['force_active'])}")
print(f"Impulse window: t={FORCE_STEP*DT:.2f}s to {(FORCE_STEP + int(IMPULSE_DURATION/DT))*DT:.2f}s")
impulse_steps = [i for i, v in enumerate(history['force_active']) if v == 1]
print(f"Impulse step indices: {impulse_steps}")

# =====================================================
# 5. Plots
# =====================================================
time_axis  = np.arange(n_steps) * DT
force_time = FORCE_STEP * DT

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig.suptitle(
    'Phase 1 — PD Controller: X+Y Tracking + Single Impulse Disturbance\n'
    f'KP=({KP_X},{KP_Y})  KD=({KD_X},{KD_Y})  '
    f'Force={F_EXTERNAL[1]:.1f}N at t={force_time:.1f}s',
    fontsize=11
)

# shade impulse window — widened for visibility
impulse_steps = [i for i, v in enumerate(history['force_active']) if v == 1]
if impulse_steps:
    t_center = time_axis[impulse_steps[0]]
    for ax in axes:
        ax.axvspan(t_center - 0.3, t_center + 0.3,
                   color='red', alpha=0.4, zorder=0, label='Impulse window')

# Panel 1 — CoM Y tracking
axes[0].plot(time_axis, history['com_goal_y'],   ls='--', color='blue',   lw=1.5, label='Goal CoM Y (sinusoid)')
axes[0].plot(time_axis, history['com_actual_y'], color='orange', lw=1.5,          label='Actual CoM Y')
axes[0].axvline(force_time, color='red', ls=':', lw=1.5, label=f'Impulse {F_EXTERNAL[1]:.1f}N')
axes[0].set_ylabel('CoM Y [m]')
axes[0].set_title('CoM Y Tracking — sinusoidal goal, controller compensates impulse')
axes[0].legend(fontsize=8)
axes[0].axhline(0, color='k', lw=0.4, ls=':')

# Panel 2 — CoM X tracking (hold at 0)
axes[1].plot(time_axis, history['com_goal_x'],   ls='--', color='steelblue', lw=1.5, label='Goal CoM X = 0')
axes[1].plot(time_axis, history['com_actual_x'], color='darkorange', lw=1.5,          label='Actual CoM X')
axes[1].axvline(force_time, color='red', ls=':', lw=1.5)
axes[1].set_ylabel('CoM X [m]')
axes[1].set_title('CoM X Tracking — hold at 0 (X gain active)')
axes[1].legend(fontsize=8)
axes[1].axhline(0, color='k', lw=0.4, ls=':')

# Panel 3 — Tracking errors X and Y
axes[2].plot(time_axis, history['error_y'], color='red',      lw=1.2, label='Error Y')
axes[2].plot(time_axis, history['error_x'], color='steelblue', lw=1.2, label='Error X')
axes[2].axhline(0, color='k', ls='--', lw=0.8)
axes[2].axvline(force_time, color='red', ls=':', lw=1.5)
axes[2].set_ylabel('Error [m]')
axes[2].set_title('Tracking Error — spike at impulse, then recovery')
axes[2].legend(fontsize=8)

# Panel 4 — Hip abd + ankle flex: PD vs fext torques
hip_c = 'Single_leg_hip_abd'
ank_c = 'Single_leg_ankle_flex'
axes[3].plot(time_axis, history['tau_pd'][hip_c],
             color='blue',      lw=1.6,          label='Hip Abd — tau_PD')
axes[3].plot(time_axis, history['tau_pd'][ank_c],
             color='green',     lw=1.6,          label='Ankle Flex — tau_PD')
axes[3].plot(time_axis, history['tau_fext'][hip_c],
             color='navy',      lw=1.2, ls='--', label='Hip Abd — tau_fext (Jt)')
axes[3].plot(time_axis, history['tau_fext'][ank_c],
             color='darkgreen', lw=1.2, ls='--', label='Ankle Flex — tau_fext (Jt)')
axes[3].axvline(force_time, color='red', ls=':', lw=1.5, label='Impulse')
axes[3].axhline(0, color='k', ls='--', lw=0.8)
axes[3].set_ylabel('Torque [Nm]')
axes[3].set_xlabel('Time [s]')
axes[3].set_title('Joint Torques: PD compensation vs impulse disturbance')
axes[3].legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig('phase1_PD_sinusoidal.png', dpi=120)
plt.close()
print("Saved -> phase1_PD_sinusoidal.png")
print("=== Phase 1 done ===")
