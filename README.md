# Example code with Pinocchio, with Gepetto viewer (Gendis' ver is with MeshCat)

## 🧠 Overview

This repository contains the necessary code to open a URDF model with Pinocchio, and replay experimental data with it.
The code focuses on : 
- Processing experimental human joint motion data (filtering, computing joints velocity and acceleration)
- Computing joint torques via inverse dynamics (RNEA function)
- Visualizing motion and center of mass (CoM) evolution in Gepetto Viewer

The system is built around the Pinocchio rigid-body dynamics library and is designed to be modular, reproducible, and extensible for future research tasks (optimal control, balance studies, assistive devices, etc.).

## 🏗️ Repository Structure
├── run_expe_data.py          # Main pipeline: data → dynamics → visualization
├── run_with_gepetto.sh       # Shell script to launch Gepetto Viewer
├── human_subject3.urdf       # Biomechanical model (URDF)
├── Data/
│   └── *.csv                 # Experimental motion capture data
└── README.md                 # This document
└── box.stl                   # Mesh for visualizing the URDF (material or texture)

## ⚙️ Dependencies & Environment
Required Python Packages
- numpy
- scipy
- pandas
- matplotlib
- pinocchio
- gepetto-viewer-corba


⚠️ Important: Gepetto Viewer must be running before executing any visualization script.

## 🚀 How to Run the Project
There is a single line to launch in a terminal to run the project : 

```bash
./run_with_gepetto.sh ./run_expe_data.py
```

### Explanations 
 ./run_with_gepetto.sh : This launches a script that launches the Gepetto Viewer server required by Pinocchio.
 ./run_expe_data.py : The script takes the name of the program and after launching Gepetto Viewer, it executes the program with Python. If another python code needs to be run with Gepetto Viewer, you can just change the name of the script here. 

 This current script (run_expe_data.py ) : 
- Loads experimental data
- Computes joint velocities & accelerations
- Applies inverse dynamics (via the rnea function)
- Displays the motion and CoM in real time

## 🔬 Detailed Pipeline Explanation
#### 1. Model Construction (URDF → Pinocchio)

- The human biomechanical model is defined in a URDF file
- Pinocchio parses:
    - Kinematic tree
    - Joint definitions
    - Inertial parameters

- Three models are built:
    - Kinematic model
    - Collision model
    - Visual model

This allows both physics-based computation and real-time visualization.

#### 2. Experimental Data Loading

- Motion capture data is stored in CSV format
- Joint angles are:
    - Selected via a configurable joint_model list
    - Converted from degrees to radians
    - Time is normalized so that the motion starts at t = 0
This makes the data consistent with Pinocchio’s internal conventions.

#### 3. Signal Processing

To obtain physically meaningful dynamics:

🔹 Low-pass filtering
- A 4th-order Butterworth filter removes measurement noise
- Cutoff frequency: 3 Hz

🔹 Numerical differentiation
- Velocities computed via finite differences
- Accelerations computed from velocities
- NaNs and infinities are explicitly handled

This step is crucial for stable inverse dynamics.

#### 4. Inverse Dynamics (RNEA)

At each timestep:
- Joint positions q
- Joint velocities dq
- Joint accelerations ddq

are passed to: pin.rnea(model, data, q, dq, ddq)


This computes the joint torques τ required to produce the observed motion.

✔ This corresponds to a pure inverse dynamics problem (no control, no feedback).

#### 5. Data Visualization (Matplotlib)

If plotting is enabled (variable plot can be set to True or False at the begining of the code):
- Joint angles
- Velocities
- Accelerations
- Torques

are plotted over time for validation and analysis.

This step is useful for:
- Detecting signal artifacts
- Comparing joints
- Exporting figures for reports

#### 6. 3D Visualization in Gepetto Viewer

The motion is replayed in real time:
- The human model is animated using joint trajectories
- A red sphere tracks the Center of Mass (CoM)
- A floor plane is added for spatial reference
- Playback speed is synchronized to 60 FPS.

############################################################################################################################################
#### 7. Additional information in the version of 

This is my addition to Oceane's original code of "./run_expe_data.py" --> ./run_expe_data_1.py
I have added components, such as 
total mass, compute CoM at neutral pose, verify joint limits, check gravity vector, compute Jacobians
Compute M(q) with crba, h(q,q̇) with rnea, implement equation of motion, add external force at the last joint (not yet CoM)