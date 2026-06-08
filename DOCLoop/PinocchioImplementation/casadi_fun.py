from math import tau

import casadi as cs
import pandas as pd
import numpy as np
from pinocchio import casadi as cpin

def make_tau_fun(cmodel):
    q  = cs.SX.sym("q",  cmodel.nq)
    dq = cs.SX.sym("dq", cmodel.nv)
    ddq = cs.SX.sym("ddq", cmodel.nv)

    cdata = cmodel.createData()

    # rnea gives tau directly
    tau = cpin.rnea(cmodel, cdata, q, dq, ddq)

    return cs.Function("tau_fun", [q, dq, ddq], [tau])

def make_com_fun(cmodel):
    q  = cs.SX.sym("q",  cmodel.nq)
    dq = cs.SX.sym("dq", cmodel.nv)
    ddq = cs.SX.sym("ddq", cmodel.nv)

    cdata = cmodel.createData()
    com = cpin.centerOfMass(cmodel, cdata, q)  # 3×1 SX
    vcom = cdata.vcom[0]  # 3×1 SX
    acom = cdata.acom[0]  # 3×1 SX
    
    return cs.Function("com_fun", [q, dq, ddq], [com, vcom, acom])

def make_pinocchio_model(cmodel, tau_fun, com_fun, N, w ):
    """
    N: number of time steps
    model: Pinocchio model (with free-flyer)
    tau_fun, com_fun: CasADi Functions built above
    """
    opti = cs.Opti()
    var = {}

    params = {}
    params['dt']   = opti.parameter(1)
    params['q0']   = opti.parameter(cmodel.nq)
    params['dq0']  = opti.parameter(cmodel.nv)
    params['goal_COM'] = opti.parameter(3)   # or 2 if planar COM

    # (Optional) other parameters like safety obstacles etc.
    var['parameters'] = params

    n = cmodel.nv  # assume nq == nv

    # ---------- Decision variables ----------
    variables = {}
    variables['q']   = opti.variable(n, N)      # q_0 ... q_{N-1}
    variables['dq']  = opti.variable(n, N-1)    # dq_0 ... dq_{N-2}
    variables['ddq'] = opti.variable(n, N-2)    # ddq_0 ... ddq_{N-3}
    var['variables'] = variables

    # ---------- Functions over trajectory ----------
    functions = {}

    q_full   = variables['q']
    dq_full  = cs.horzcat(variables['dq'],  cs.DM.zeros(n,1))  # pad with zero at final step
    ddq_full = cs.horzcat(variables['ddq'], cs.DM.zeros(n,2))  # pad with zeros at last 2 steps

    # Compute tau for each k
    tau_list   = []
    com_list   = []

    for k in range(N):
        qk   = q_full[:, k]
        dqk  = dq_full[:, k]
        ddqk = ddq_full[:, k]

        tau_k  = tau_fun(qk, dqk, ddqk)   # (n,)
        com_k  = com_fun(qk)    # (3,)

        tau_list.append(tau_k)
        com_list.append(com_k)

    functions['model_tau'] = cs.horzcat(*tau_list)      # (n, N)
    functions['COM']       = cs.horzcat(*com_list)      # (3, N)

    dt = params['dt']
    com_traj = functions['COM']                          # (3, N)
    # Central-difference acceleration: (COM[:,k+1] - 2*COM[:,k] + COM[:,k-1]) / dt²
    acom_inner = (com_traj[:, 2:] - 2*com_traj[:, 1:-1] + com_traj[:, :-2]) / dt**2  # (3, N-2)
    # Pad edges so shape stays (3, N)
    functions['aCOM'] = cs.horzcat(acom_inner[:, 0],    # repeat first
                                   acom_inner,
                                   acom_inner[:, -1])   # repeat last
    var['functions'] = functions


    # ---------- Constraints ----------
    constraints = {}

    # initial conditions
    constraints['initial_pos'] = variables['q'][:, 0]   - params['q0']
    constraints['initial_vel'] = variables['dq'][:, 0]  - params['dq0']

    # discrete dynamics (Euler)
    constraints['dynamics_pos'] = (variables['q'][:,1:] - variables['q'][:,:-1]
                                   - variables['dq'] * params['dt'])
    constraints['dynamics_vel'] = (variables['dq'][:,1:] - variables['dq'][:,:-1]
                                   - variables['ddq'] * params['dt'])

    # COM constraints
    com = functions['COM']  # (3,N)
    constraints['com_final']= com[:, -1]  - params['goal_COM']  # or COM_goal param
    

   
    # Add to Opti
    opti.subject_to(constraints['initial_pos'] == 0)
    opti.subject_to(constraints['initial_vel'] == 0)
    opti.subject_to(constraints['dynamics_pos'] == 0)
    opti.subject_to(constraints['dynamics_vel'] == 0)
    opti.subject_to(constraints['com_final']  == 0)

    # opti.subject_to(opti.bounded(-100, variables['dq'], 100))
    # opti.subject_to(opti.bounded(-100, functions['model_tau'], 100))
    constraints['final_vel'] = variables['dq'][:, -1]
    opti.subject_to(constraints['final_vel'] == 0)
    opti.subject_to(opti.bounded(-2, variables['q'], 2))

    var['constraints'] = constraints

        #Costs
    costs = {}
    tau  = functions['model_tau']   # (n, N)
    acom = functions['aCOM']        # (3, N)

    # Energy: minimize squared torques
    tau_max = 10.0
    costs['energy_cost'] = cs.sumsqr(tau) / N / tau_max**2

    # COM jerk: penalize rate of change of COM acceleration
    com_jerk = acom[:, 1:] - acom[:, :-1]          # (3, N-1)
    costs['com_jerk'] = cs.sumsqr(com_jerk) / N

    # Torque jerk: penalize sudden changes in torque per joint
    tau_jerk = tau[:, 1:] - tau[:, :-1]             # (n, N-1)
    costs['tau_jerk'] = cs.sumsqr(tau_jerk) / N / tau_max**2

    # Joint acceleration jerk: penalize rate of change of ddq per joint
    ddq_jerk = variables['ddq'][:, 1:] - variables['ddq'][:, :-1]   # (n, N-3)
    costs['joint_jerk'] = cs.sumsqr(ddq_jerk) / N

    var['costs'] = costs

    # Total cost
    w_energy     = w[1]
    w_com_jerk   = w[2] if len(w) > 2 else 1.0
    w_tau_jerk   = w[3] if len(w) > 3 else 0.1
    w_joint_jerk = w[4] if len(w) > 4 else 0.1

    J = (w_energy     * costs['energy_cost']
        + w_com_jerk   * costs['com_jerk']
        + w_tau_jerk   * costs['tau_jerk']
        + w_joint_jerk * costs['joint_jerk'])

    opti.minimize(J)

    return opti, var



def instantiate_pinocchio_model(var, opti, dt, q0, dq0, goal_COM, q_guess, dq_guess, ddq_guess):
    
    p = var['parameters']
    opti.set_value(p['dt'], dt)
    opti.set_value(p['q0'], q0)
    opti.set_value(p['dq0'], dq0)
    opti.set_value(p['goal_COM'], goal_COM)

    v = var['variables']
    opti.set_initial(v['q'],   q_guess)
    opti.set_initial(v['dq'],  dq_guess)
    opti.set_initial(v['ddq'], ddq_guess)



def numerize_var(model_var, opti, initial_flag=False):
    """
    Evaluate all symbolic CasADi variables/parameters/functions in a model
    into numeric CasADi DM arrays, either at the current solution or at
    the initial guess.

    Args:
        model_var (dict): The structure returned by make_ndof_model().
        opti (casadi.Opti): The CasADi Opti instance.
        initial_flag (bool, optional): 
            If True, evaluate using opti.initial(). 
            Otherwise, use the optimized values. Default: False.

    Returns:
        dict: A dictionary 'num_var' with the same structure as model_var,
              but all CasADi symbols replaced by numeric DM values.
    """

    num_var = {}

    # Loop over main categories: 'variables', 'parameters', 'functions'
    for category_name, category_content in model_var.items():
        num_var[category_name] = {}

        # Loop over computables within each category
        for computable_name, computable_value in category_content.items():
            # Case 1: list (cell array in MATLAB)
            if isinstance(computable_value, (list, tuple)):
                num_var[category_name][computable_name] = []
                for item in computable_value:
                    if not initial_flag:
                        num_var[category_name][computable_name].append(opti.value(item))
                    else:
                        num_var[category_name][computable_name].append(
                            opti.value(item, opti.initial())
                        )

            # Case 2: direct symbolic expression (MX/DM)
            else:
                if not initial_flag:
                    num_var[category_name][computable_name] = opti.value(computable_value)
                else:
                    num_var[category_name][computable_name] = opti.value(
                        computable_value, opti.initial()
                    )

    return num_var