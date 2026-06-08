import time
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf
import matplotlib.pyplot as plt


def play_in_meshcat(model, collision_model, visual_model, q_trajectory,
                    goal_com, com_traj, root, dt=0.02, window_name="OCP"):
    """
    Play a joint configuration trajectory in Meshcat Viewer.
    q_trajectory : (nq, T)
    com_traj     : (3,  T)
    goal_com     : (3,)
    """
    from pinocchio.visualize import MeshcatVisualizer

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(open=True, loadModel=True)
    except Exception as e:
        print("Could not initialize Meshcat Viewer.")
        raise e

    time.sleep(1.5)   # let the viewer load

    # Goal COM — green sphere (static)
    viz.viewer['goal_com'].set_object(
        g.Sphere(0.06),
        g.MeshLambertMaterial(color=0x00ff00)
    )
    viz.viewer['goal_com'].set_transform(
        tf.translation_matrix([float(goal_com[0]),
                                float(goal_com[1]),
                                float(goal_com[2])])
    )

    # Current COM — yellow sphere (animated)
    viz.viewer['com'].set_object(
        g.Sphere(0.05),
        g.MeshLambertMaterial(color=0xffff00)
    )

    # Ensure shape (nq, T)
    q_trajectory = np.array(q_trajectory)
    if q_trajectory.ndim == 1:
        q_trajectory = q_trajectory.reshape(model.nq, -1)
    elif q_trajectory.shape[0] != model.nq:
        q_trajectory = q_trajectory.T

    print("\n▶ Playing optimized motion in Meshcat Viewer...\n")

    for k in range(q_trajectory.shape[1]):
        viz.display(q_trajectory[:, k])

        com_k = com_traj[:, k]
        viz.viewer['com'].set_transform(
            tf.translation_matrix([float(com_k[0]),
                                   float(com_k[1]),
                                   float(com_k[2])])
        )
        time.sleep(dt)

    print("\n✓ Playback finished.\n")


def add_com_sphere(viz, com_radius=0.05, color=0xffff00):
    """Add animated COM sphere (yellow)."""
    viz.viewer['com'].set_object(
        g.Sphere(com_radius),
        g.MeshLambertMaterial(color=color)
    )
    return 'com'


def add_static_sphere(viz, name, position, radius=0.06, color=0x00ff00):
    """Add a static sphere at a fixed world position."""
    viz.viewer[name].set_object(
        g.Sphere(radius),
        g.MeshLambertMaterial(color=color)
    )
    viz.viewer[name].set_transform(
        tf.translation_matrix([float(position[0]),
                                float(position[1]),
                                float(position[2])])
    )
    return name
    
def plot_results(t, q_arr, dq_arr, ddq_arr, tau_arr, com_arr, com_goal):
    """
    Plot joint trajectories, velocities, accelerations, torques, and COM.
    All arrays shape: (n_dof, T) except com_arr (3, T) and com_goal (3,).
    """
    nq = q_arr.shape[0]
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)

    # --- Joint positions ---
    ax = axes[0]
    for i in range(nq):
        ax.plot(t, q_arr[i, :len(t)], label=f'q{i+1}')
    ax.set_ylabel('Joint pos [rad]')
    ax.legend(); ax.grid(True)

    # --- Joint velocities ---
    ax = axes[1]
    t_dq = t[:dq_arr.shape[1]]
    for i in range(nq):
        ax.plot(t_dq, dq_arr[i, :len(t_dq)], label=f'dq{i+1}')
    ax.set_ylabel('Joint vel [rad/s]')
    ax.legend(); ax.grid(True)

    # --- Joint accelerations ---
    ax = axes[2]
    t_ddq = t[:ddq_arr.shape[1]]
    for i in range(nq):
        ax.plot(t_ddq, ddq_arr[i, :len(t_ddq)], label=f'ddq{i+1}')
    ax.set_ylabel('Joint acc [rad/s²]')
    ax.legend(); ax.grid(True)

    # --- Torques ---
    ax = axes[3]
    t_tau = t[:tau_arr.shape[1]]
    for i in range(tau_arr.shape[0]):
        ax.plot(t_tau, tau_arr[i, :len(t_tau)], label=f'tau{i+1}')
    ax.set_ylabel('Torque [Nm]')
    ax.legend(); ax.grid(True)

    # --- COM trajectory ---
    ax = axes[4]
    labels = ['x', 'y', 'z']
    t_com = t[:com_arr.shape[1]]
    for i in range(3):
        ax.plot(t_com, com_arr[i, :len(t_com)], label=f'COM {labels[i]}')
        ax.axhline(com_goal[i], linestyle='--', alpha=0.5, label=f'goal {labels[i]}')
    ax.set_ylabel('COM [m]')
    ax.set_xlabel('Time [s]')
    ax.legend(); ax.grid(True)

    fig.suptitle('OCP Solution', fontsize=14)
    plt.tight_layout()
    plt.show()