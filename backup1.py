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


# =====================================================
# 3. Plots
# =====================================================

if plot:
    df_plot = df_filtered.iloc[::10]   # downsample here are free to choose, but for this large dataset, I took the liberty to downsample by 10 for faster plotting. Adjust as needed.

    def save_fig(data_cols, title, filename):
        """Plot columns against time, save PNG, close to free memory."""
        fig, ax = plt.subplots(figsize=(10, 4))
        for col in data_cols:
            ax.plot(df_plot['time'], df_plot[col], label=col)   # ← explicit x=time fixes straight lines
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")                  # fixed location — avoids slow "best" scan
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)                                # release memory immediately
        print(f"  Saved: {filename}")

    print("=== Saving plots ===")
    save_fig(joint_model,                            "Joint Angles (rad)",         "plot_angles.png")
    save_fig(['vel_' + c for c in joint_model],      "Joint Velocities (rad/s)",   "plot_velocities.png")
    save_fig(['acc_' + c for c in joint_model],      "Joint Accelerations (r/s²)", "plot_accelerations.png")
    save_fig(['tau_' + c for c in joint_model],      "Joint Torques (Nm)",         "plot_torques.png")
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

try:
    viz.initViewer(open=True)          # opens browser automatically
except ImportError as err:
    print(f"MeshCat import error: {err}\nInstall with: pip install meshcat")
    sys.exit(0)

time_module.sleep(2)                           # wait for server to be ready before loading model
viz.loadViewerModel(rootNodeName=root)     # ← pass mesh_dir so textures/meshes load correctly
print("MeshCat open at: http://127.0.0.1:7000/static/")

q0 = pin.neutral(model)
viz.display(q0)                                # show skeleton at neutral pose first

rate = 1 / 60                                  # ~60 fps animation

for i in range(len(df_filtered)):
    q = df_filtered[joint_model].iloc[i].values.astype(float)
    viz.display(q)                             # update skeleton pose each frame

    com = pin.centerOfMass(model, data, q)
    viz.viewer['com'].set_object(              # draw CoM as a red sphere
        g.Sphere(0.08),
        g.MeshLambertMaterial(color=0xff0000))
    viz.viewer['com'].set_transform(
        tf.translation_matrix([com[0], com[1], com[2]]))

    time_module.sleep(rate)

#plt.show()
# ── Clean shutdown ────────────────────────────────────────────────────────────
viz.viewer.close()     # close MeshCat server
del viz
gc.collect()           # force memory release
print("=== Done ===")


# =====================================================
# 1. Build Pinocchio model
# =====================================================
mesh_dir = os.path.dirname(os.path.abspath(__file__))
urdf_filename = "human_subject3.urdf"
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

# Jacobian of last joint
pin.computeJointJacobians(model, data, q0)
J = pin.getJointJacobian(model, data, model.njoints-1, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

# M(q)
M = pin.crba(model, data, q0)

# h(q, qdot)
h = pin.rnea(model, data, q0, v0, a0)

# Equation of motion
tau_eom = M @ a0 + h

# ── External force at last joint ──────────────────────
fext = [pin.Force.Zero() for _ in range(model.njoints)]

joint_id = model.njoints - 1          # last joint for now

F_local = np.array([0.0, 0.0, 10.0]) # 10N in Z, joint local frame
M_local = np.array([0.0, 0.0, 0.0])  # no moment for now

fext[joint_id] = pin.Force(F_local, M_local)

tau_with_fext = pin.rnea(model, data, q0, v0, a0, fext)
# ─────────────────────────────────────────────────────

# Save to file
with open("model_info.txt", "w") as f:
    f.write(f"Total mass: {total_mass}\n")
    f.write(f"CoM at neutral: {com0}\n")
    f.write("Joint limits:\n")
    for i in range(1, model.njoints):
        name = model.names[i]
        lower = model.lowerPositionLimit[i-1]
        upper = model.upperPositionLimit[i-1]
        f.write(f"  Joint {name}: [{lower:.3f}, {upper:.3f}] rad\n")
    f.write(f"Gravity: {model.gravity.linear}\n")
    f.write(f"Jacobian of last joint:\n{J}\n")
    f.write(f"Mass matrix M(q):\n{M}\n")
    f.write(f"h(q, qdot): {h}\n")
    f.write(f"tau_eom: {tau_eom}\n")
    f.write(f"tau_with_fext: {tau_with_fext}\n")

# =====================================================
# 2. CSV / MeshCat section continues below
# =====================================================

# Step 1 — create zero force for every joint
fext = [pin.Force.Zero() for _ in range(model.njoints)]

# Step 2 — pick the joint
joint_id = model.getJointId("joint_name")

# Step 3 — define force and moment in joint local frame
F_local = np.array([0.0, 0.0, 10.0])  # N
M_local = np.array([0.0, 0.0, 0.0])   # Nm

# Step 4 — apply it
fext[joint_id] = pin.Force(F_local, M_local)

# Step 5 — run RNEA
tau = pin.rnea(model, data, q0, v0, a0, fext)



#====================================================================================
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

try:
    viz.initViewer(open=True)          # opens browser automatically
except ImportError as err:
    print(f"MeshCat import error: {err}\nInstall with: pip install meshcat")
    sys.exit(0)

time_module.sleep(2)                           # wait for server to be ready before loading model
viz.loadViewerModel(rootNodeName=root)     # ← pass mesh_dir so textures/meshes load correctly
print("MeshCat open at: http://127.0.0.1:7000/static/")

q0 = pin.neutral(model)
viz.display(q0)                                # show skeleton at neutral pose first

rate = 1 / 60                                  # ~60 fps animation

for i in range(len(df_filtered)):
    q = df_filtered[joint_model].iloc[i].values.astype(float)
    viz.display(q)                             # update skeleton pose each frame

    com = pin.centerOfMass(model, data, q)
    viz.viewer['com'].set_object(              # draw CoM as a red sphere
        g.Sphere(0.08),
        g.MeshLambertMaterial(color=0xff0000))
    viz.viewer['com'].set_transform(
        tf.translation_matrix([com[0], com[1], com[2]]))

    time_module.sleep(rate)

#plt.show()
# ── Clean shutdown ────────────────────────────────────────────────────────────
del viz
gc.collect()           # force memory release
print("=== Done ===")






print("Columns in CSV:", df.columns.tolist())


missing = [c for c in all_joints if c not in df.columns]
print(f"Missing from CSV ({len(missing)}): {missing}")

if joint_model:
    df[joint_model] = np.deg2rad(df[joint_model].astype(float))
else:
    print("No joint angle columns found to convert to radians.")

df['time'] = df['marker_timestamp'] - df['marker_timestamp'].iloc[0]

# ── Build df_filtered with .values to break pandas copy link ─────────────
df_filtered = pd.DataFrame()
df_filtered['time'] = df['time'].values

# ── Copy raw angles into df_filtered ──────────────────────────────────────
for col in joint_model:
    df_filtered[col] = df[col].values

# ── Define filter coefficients HERE, before using them ───────────────────
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

# ── Diagnosis — check before plotting ────────────────────────────────────
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
    save_fig(['com_x', 'com_y', 'com_z'],         "CoM Position (m)",            "plot_CoM.png")
    print("=== All plots saved as PNG ===")