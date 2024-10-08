import pandas as pd
import numpy as np

def rounding_to_integer(list_with_solution):
    """
        Rounds x solutions to integer.
    """
    return [round(x) for x in list_with_solution]

def post_run_perfromance(f_stars, feasibility_flags, skipped):
    """
        Returns the count of solved ILPs, infeasible ILPs, Pareto optimal solutions found, non-Pareto optimal solutions 
        (if any), and skipped relaxations during the solving process.
    """
    total_solved = len(f_stars)
    total_infeasible_solutions = len([x for x in feasibility_flags if x!=2])
    unique_pareto_solutions, unique_non_pareto_solutions = pareto_check(f_stars)

    return total_solved, total_infeasible_solutions, unique_pareto_solutions, unique_non_pareto_solutions, sum(skipped)

def pareto_check(f_stars):
    """
        Checks whether the set of solutions (images) are indeed Pareto-optimal.
    """
    temp_df = pd.DataFrame(f_stars).drop_duplicates().dropna()
    a = list(temp_df.itertuples(index=False,name=None))

    pareto = []
    not_pareto = []

    for t in a:
        temp_list = [x for x in a if np.all(np.array(x)>=np.array(t)) and x!=t]
        
        if temp_list:
            not_pareto.append(t)
        else:
            pareto.append(t)

    return len(pareto), len(not_pareto)#, pareto, not_pareto, 

def problem_unpacking_sc(file):
    """
        Unpacks the set covering input files into the required format and adjust them for maximization.
    """
    file_data = [i.strip().split() for i in open(file).readlines()]
    temp = file_data.pop(0)
    n_decision_var = int(temp[0])
    n_constraints = int(temp[1])

    temp = file_data.pop(0)
    n_objectives = int(temp[0])

    file_data = [[int(y) for y in x] for x in file_data]
    objective_matrix = file_data[:n_objectives]
    objective_matrix = np.array(objective_matrix)*(-1)
    constraint_matrix_index = file_data[n_objectives:]
    constraint_matrix_index = [constraint_matrix_index[i] for i in range(1,len(constraint_matrix_index),2)]
    constraint_matrix = np.zeros((n_constraints, n_decision_var))

    for i in range(n_constraints):
        for v in constraint_matrix_index[i]:
            constraint_matrix[i][v-1] +=1

    b_vector = np.ones(n_constraints)

    return objective_matrix, constraint_matrix, b_vector, n_objectives

def problem_unpacking_kp(file):
    """
        Unpacks the knapsack input files into the required format.
    """

    file_data = [i.strip().split() for i in open(file).readlines()]
    temp = file_data.pop(0)
    n_objectives = int(temp[-1])

    objective_matrix = [[int(y) for y in x] for x in file_data]
    b_vector = objective_matrix.pop(-1)
    constraint_matrix = [objective_matrix.pop(-1)]
    constraint_matrix = np.array(constraint_matrix)
    b_vector = np.array(b_vector)

    return objective_matrix, constraint_matrix, b_vector, n_objectives

def relaxation_check(f_stars, ef_vectors, feasibility_flags, f_columns, ef_columns):
    """
        After Phase I, the sets of eps values, objective values, and feasibility flags are cleaned and reduced by 
        excluding the relaxation problems. This refined data is then used in Phase II, Block E.
    """
    df = pd.concat([pd.DataFrame(ef_vectors, columns=ef_columns), 
                    pd.DataFrame(f_stars, columns=f_columns),
                    pd.DataFrame(feasibility_flags, columns=['flag'])], axis=1).drop_duplicates()
    
    df['id'] = df.groupby(f_columns).grouper.group_info[0]

    ef_vectors_p = []
    f_stars_p = []
    feasibility_flags_p = []

    for group_id in df['id'].unique():
        temp_df = df[(df['id']==group_id)]

        a = list(temp_df[ef_columns].itertuples(index=False,name=None))
        
        temp_flag = df[(df['id']==group_id)]['flag'].to_list()[0]
        temp_f_star = df[(df['id']==group_id)][f_columns].drop_duplicates().to_numpy()[0]

        for t in a:
            temp_list = [x for x in a if np.all(np.array(x)<=np.array(t)) and x!=t]
            
            if not temp_list:
                ef_vectors_p.append(np.array(t))
                f_stars_p.append(temp_f_star)
                feasibility_flags_p.append(temp_flag)
    
    return ef_vectors_p, f_stars_p, feasibility_flags_p, len(df['id'].unique())


    