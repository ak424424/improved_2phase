import numpy as np
# import jax.numpy as np
import pandas as pd
import itertools
import gurobipy as gp
from gurobipy import GRB

def ef_vs_efs_check_vs_f_stars_check(ef, ef_vectors, feasibility_flags, f_stars):
    ef = np.array(ef)
    ef_vectors = np.array(ef_vectors)
    feasibility_flags = np.array(feasibility_flags)
    f_stars = np.array(f_stars)
    
    # Vectorized computation for the first filter
    mask = np.all(ef >= ef_vectors, axis=1)
    # filtered_vectors = ef_vectors[mask]
    filtered_f_stars = f_stars[mask]
    filtered_flags = feasibility_flags[mask]
    
    # Process the feasible and not feasible lists
    feasible_mask = (filtered_flags == 2) & np.all(ef <= filtered_f_stars[:, :-1], axis=1)
    temp_list_f_stars_feasible = filtered_f_stars[feasible_mask]
    not_feasible_mask = (filtered_flags == 3)
    temp_list_f_stars_not_feasible = filtered_f_stars[not_feasible_mask]
    
    # Return based on the findings
    if temp_list_f_stars_feasible.size>0:
        return temp_list_f_stars_feasible[0], 2, 1
    elif temp_list_f_stars_not_feasible.size>0:
        return [], 3, 1
    else:
        return -1, -1, -1
  
def objective_coefficients_matrix(f_min, f_max, eps, index_main_objective, objective_matrix):
    # 1. Calculate adjustment coefficient for each objective function
    ranges_eps = [eps/(x-y) for x,y in zip(f_max, f_min)]
    ranges_eps.insert(index_main_objective, 1)
    # 2. Apply adjustment coefficient
    adjusted_objective_matrix = np.array([x*y for x,y in zip(objective_matrix,ranges_eps)])

    return adjusted_objective_matrix

def fmin_fmax_estimation(other_objective_matrix, constraint_matrix, b_vector, n_decision_var, OutputFlag = 0):

    f_max = []
    f_min = []

    # Create a new model
    model = gp.Model('min_max')
    model.setParam('OutputFlag', OutputFlag)
    # Define decision variables
    x = model.addMVar(n_decision_var, vtype=GRB.BINARY, name="X")
    # Define constraints
    model.addConstr(np.array(constraint_matrix)@ x <=b_vector)

    for obj in other_objective_matrix:
        objective = np.array(obj)@ x
        model.setObjective(objective, GRB.MAXIMIZE)
        model.optimize()
        f_max.append(model.ObjVal)

    for obj in other_objective_matrix:
        objective = np.array(obj)@ x
        model.setObjective(objective, GRB.MINIMIZE)

        model.optimize()
        f_min.append(model.ObjVal)

    return f_max, f_min

def acceleration_algo_with_bouncing_steps(f_star, f_rwv, index_main_objective):
    f_rwv = f_rwv[:1] + [min(x,y) for x,y in zip(f_star[1:index_main_objective], f_rwv[1:])]
    return f_rwv

def acceleration_algo_with_early_exit(ef, f_min, f_max, n_objectives):
    last_answer = 'yes'
    i = 0

    while last_answer == 'yes' and i<(n_objectives-2):
        if ef[i] == f_min[i]:
            i = i+1
        else:
            ef = f_max[:i+1] + ef[i+1:]
            last_answer = 'no'
            i = i+1
    
    if i == (n_objectives-2) and last_answer == 'yes':
        return f_max
    else: 
        return ef  

def main_check(ef, f_min, f_max, f_rwv, n_objectives):
    # 1. The first check ~ f_1
    termination_flag = 0
    i = 0
    if ef[i] < f_max[i]:
         ef[i] = ef[i]+1
         return f_rwv, ef, termination_flag
    else:
        # 2. If the first check is not passed, the following loop is srated
        ef[i] = f_min[i]-1
        i = i+1 
        last_answer = 'no'
    
        while last_answer == 'no' and i<=(n_objectives-2):
            if ef[i] < f_max[i]:        # mini check 1                
                ef[i] = f_rwv[i]
                f_rwv[i] = f_max[i]

                if ef[i] < f_max[i]:    # mini check 2
                    addition_ef = [1]*(i+1) + [0]*(n_objectives-2-i)
                    ef = [sum(x) for x in zip(ef, addition_ef)]
                    f_rwv = f_rwv[:1] + f_max[1:i] + f_rwv[i:]
                    last_answer = 'yes' 
                    i = i+1
                
                else:
                    ef[i] = f_min[0]-1
                    if i == (n_objectives-2):
                        termination_flag = 1
                    i = i+1                

            else:
                ef[i] = f_min[0]-1
                if i == (n_objectives-2):
                    termination_flag = 1
                i = i+1
        
        return f_rwv, ef, termination_flag
    
def lexi_fmin_fmax_estimation(objective_matrix, constraint_matrix, b_vector, n_decision_var, OutputFlag = 0):
    f_max = []
    f_min = []

    # Create a new model
    model = gp.Model('min_max')
    model.setParam('OutputFlag', OutputFlag)
    # Define decision variables
    x = model.addMVar(n_decision_var, vtype=GRB.BINARY, name="X")
    # Define constraints
    model.addConstr(np.array(constraint_matrix)@ x <=b_vector)

    for obj in objective_matrix:
        objective = np.array(obj)@ x
        model.setObjective(objective, GRB.MAXIMIZE)
        model.optimize()
        f_max.append(model.ObjVal)

    # 2. Nadir points
    payoff_matrix = []

    for obj in objective_matrix:
        # 2.0 Technical part
        obj_order = list(objective_matrix).index(obj)
        temp_objectives = list(objective_matrix).copy()
        temp_objectives.insert(0, temp_objectives.pop(obj_order))
        temp_payoff_row = [f_max[obj_order]]

        # 2.1 Create a new model
        model = gp.Model('nadir_values')
        model.setParam('OutputFlag', OutputFlag)
        # Define decision variables
        x = model.addMVar(n_decision_var, vtype=GRB.BINARY, name="X")
        # Define constraints
        model.addConstr(np.array(constraint_matrix) @ x <=  b_vector)

        while len(temp_objectives)>1:
            model.addConstr(np.array(temp_objectives.pop(0)) @ x >= temp_payoff_row[-1])
            objective = np.array(temp_objectives[0])@ x
            model.setObjective(objective, GRB.MAXIMIZE)
            model.optimize()
            temp_payoff_row.append(model.ObjVal)

        # 2.2 Ordering
        temp_payoff_row.insert(obj_order, temp_payoff_row.pop(0))
        payoff_matrix.append(temp_payoff_row)

    # eps vectors
    # 1. Define lower bound of range

    for i in range(0, len(objective_matrix)):
        # 1. Define range
        temp_list = [x[i] for x in payoff_matrix]
        temp_lb = min(temp_list)
        f_min.append(temp_lb)


    return f_max, f_min

def updated_ranges(f_stars, n_obj):
    fs = ['f'+str(x) for x in range(1,n_obj+1)]

    temp_df = pd.DataFrame(f_stars, columns=fs).dropna()
    temp_df = temp_df[temp_df['f1']!=-1]
    
    updated_f_min = []
    updated_f_max = []

    for f in fs:
        updated_f_min.append(temp_df[f].min())
        updated_f_max.append(temp_df[f].max())

    return updated_f_max, updated_f_min

def main_check_improved(ef, f_min, f_max, f_rwv, n_objectives, jump):
    # 1. The first check ~ f_1
    termination_flag = 0
    i = 0
    if ef[i] < f_max[i]:
         ef[i] = ef[i]+1 
         return f_rwv, ef, termination_flag
    else:
        # 2. If the first check is not passed, the following loop is srated
        ef[i] = f_min[i]-1
        i = i+1 
        last_answer = 'no'
    
        while last_answer == 'no' and i<=(n_objectives-2):
            if ef[i] < f_max[i]:        # mini check 1
                
                ef[i] = f_rwv[i]
                f_rwv[i] = f_max[i]

                if ef[i] < f_max[i]:    # mini check 2
                    addition_ef = jump[:i+1] + [0]*(n_objectives-2-i)
                    ef = [sum(x) for x in zip(ef, addition_ef)]
                    f_rwv = f_rwv[:1] + f_max[1:i] + f_rwv[i:]
                    last_answer = 'yes' 
                    i = i+1
                
                else:
                    ef[i] = f_min[0]-1
                    if i == (n_objectives-2):
                        termination_flag = 1
                    i = i+1                

            else:
                ef[i] = f_min[0]-1
                if i == (n_objectives-2):
                    termination_flag = 1
                i = i+1
        
        return f_rwv, ef, termination_flag
    
def main_check_loop_flag(ef, f_min, f_max, f_rwv, n_objectives, jump):
    # 1. The first check ~ f_1
    termination_flag = 0
    i = 0
    if ef[i] < f_max[i]:
         ef[i] = ef[i]+1 
         return f_rwv, ef, termination_flag, 0
    else:
        # 2. If the first check is not passed, the following loop is srated
        ef[i] = f_min[i]-1
        i = i+1 
        last_answer = 'no'
    
        while last_answer == 'no' and i<=(n_objectives-2):
            if ef[i] < f_max[i]:        # mini check 1
                
                ef[i] = f_rwv[i]
                f_rwv[i] = f_max[i]

                if ef[i] < f_max[i]:    # mini check 2
                    addition_ef = jump[:i+1] + [0]*(n_objectives-2-i)
                    ef = [sum(x) for x in zip(ef, addition_ef)]
                    f_rwv = f_rwv[:1] + f_max[1:i] + f_rwv[i:]
                    last_answer = 'yes' 
                    i = i+1
                
                else:
                    ef[i] = f_min[0]-1
                    if i == (n_objectives-2):
                        termination_flag = 1
                    i = i+1                

            else:
                ef[i] = f_min[0]-1
                if i == (n_objectives-2):
                    termination_flag = 1
                i = i+1
        
        return f_rwv, ef, termination_flag, 1