import gurobipy as gp
from gurobipy import GRB
from saugmecon_utils import *
from utils import *
from utils_ML import *

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import datetime
import random
from math import prod

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, silhouette_score, 
                             davies_bouldin_score, accuracy_score, f1_score)


class NumericalStudy:
    """
        The consolidated framework
    """

    def __init__(self, file, train_orders_number = 5, train_runs_per_order = 100, 
                 train_random_jump_lb = 10, train_random_jump_ub = 100, search_space_regions = 4, 
                 timeout=1200, k_range=[2,3,4,5], min_sample_size_per_cluster = 5, quantile=0.5,
                 problem_type = 'knapsack') -> None:
        self.file = file
        self.problem_type = problem_type
        self.train_orders_number = train_orders_number                  # t
        self.train_runs_per_order = train_runs_per_order                # C
        self.train_random_jump_lb = train_random_jump_lb                # j_lb
        self.train_random_jump_ub = train_random_jump_ub                # j_ub
        self.search_space_regions = search_space_regions                # the default number of clusters (k) for Block C
        self.timeout = timeout                                          # time limit of each experiment
        self.k_range = k_range                                          # the range of k for Block C
        self.min_sample_size_per_cluster = min_sample_size_per_cluster  # minimal number of observations to be included into each clusters
        self.clustering_fail_flag = 0                                   # if there is at least one cluster with less observations than minimal allowed number, the flag is 1
        self.quantile = quantile                                        # quantile for computing step size (Algorithm 2)

        # The tables to record results
        self.results_columns = ['p','n','Combination', '# of iterations', '# of solved IPs', 
                                '# of skipped IPs (relaxations)', 
                                '# of infeasible IPs', '# of Pareto efficient solutions',  'Total time in seconds', 
                                'Total solver time in seconds', 'Scenario']
        
        self.df_results_final = pd.DataFrame(columns=self.results_columns)

        self.train_info_columns = ['p','n','Combination', 'Train runs, sec', 'ML 1, sec', 'ML 2, sec',
                                   'MAE', 'RMSE', 'Silhouette Score', 'Davies-Bouldin Index', 'Accuracy', 'macro F1 score',
                                   'Scenario']
        
        self.df_train_info_final = pd.DataFrame(columns=self.train_info_columns)

        self.time_train_runs = 0
        self.time_ml_model_1 = 0
        self.time_ml_model_2 = 0

        self.mae = None
        self.rmse = None
        self.ss = None
        self.dbi = None
        self.accuracy = None
        self.f1_macro = None

        self.jump_updated_general = []
        self.jumps_updated_regions ={}
        self.best_combo = ()
        self.df_best_combo = None
        self.confusion_matrix_ml_2 = None

    def file_unpacking(self):
        """
        Unpacks input file.
        """
        if self.problem_type == 'knapsack':
            objective_matrix_given, constraint_matrix, b_vector, p = problem_unpacking_kp(self.file)
            self.f_star_coef = 1
        elif self.problem_type == 'set_covering':
            objective_matrix_given, constraint_matrix, b_vector, p = problem_unpacking_sc(self.file)
            self.f_star_coef = -1

        self.p = p
        self.n = len(objective_matrix_given[0])
        self.given_order = tuple((i for i in range(p)))
        self.all_orders = list(itertools.permutations(self.given_order, p))
        self.train_orders = random.sample(self.all_orders , k=self.train_orders_number)
        self.objective_matrix_given = objective_matrix_given.copy()
        self.constraint_matrix = np.array(constraint_matrix)
        self.b_vector = np.array(b_vector)

        self.f_columns  = ['f'+str(x) for x in range(1,self.p+1)]
        self.ef_columns = ['ef'+str(x) for x in range(1, self.p+1)]

    def train_runs(self):
        """
            Phase I. Algorithm 1
        """
        print('==== Train runs ====')
        x_solutions_train = []
        for_graph_ef_vectors = []
        for_graph_f_stars = []
        for_graph_feasibility_flag = []

        train_run_times = []
        train_orders = []

        for test_order in self.train_orders:                            
            objective_matrix = self.objective_matrix_given.copy()
            objective_matrix = [objective_matrix[i] for i in test_order]
            objective_matrix = np.array(objective_matrix)

            # Select main objective 
            n_decision_var = self.n
            n_objectives = self.p
            index_main_objective = n_objectives - 1
            eps = 1

            other_objective_matrix = np.delete(objective_matrix, index_main_objective, axis=0)

            # Initialize loop_control variables
            f_max, f_min = fmin_fmax_estimation(other_objective_matrix, self.constraint_matrix, 
                                                self.b_vector, n_decision_var)
            f_rwv = f_max.copy()#[-1] + f_max[1:]
            ef = f_min.copy()

            # help things
            adjusted_objective_matrix = objective_coefficients_matrix(f_min, f_max, eps, index_main_objective, objective_matrix)
            termination_flag = 0                                        # 1 - end the search
            run = 1

            # Initialize model
            main_model = gp.Model('SAUGMECON')
            main_model.setParam('OutputFlag', 0)
            main_model.setParam('DualReductions', 0)

            # Define decision variables
            x = main_model.addMVar(n_decision_var, vtype=GRB.BINARY, name="X")
            # Set objective
            objective = np.sum(adjusted_objective_matrix, axis=0) @ x
            main_model.setObjective(objective, GRB.MAXIMIZE)
            # Define constraints
            if self.problem_type == 'knapsack':
                main_model.addConstr(self.constraint_matrix@ x <= self.b_vector)
            elif self.problem_type == 'set_covering':
                main_model.addConstr(self.constraint_matrix@ x >= self.b_vector)
            
            obj_constr = main_model.addConstr(other_objective_matrix@ x >= np.array(ef))
            main_model.update()

            start_time = datetime.datetime.now()

            f_star_inf = np.array([-1]*n_objectives)
            ef_vectors = [np.array([-1]*(n_objectives-1))]
            f_stars = [f_star_inf]
            feasibility_flags = [-1]

            correct_order = [test_order.index(i) for i in self.given_order]
            
            while termination_flag != 1 and run<=self.train_runs_per_order:
                start_time_run = datetime.datetime.now()
                jump = [random.choice(range(self.train_random_jump_lb,self.train_random_jump_ub)) for _ in range(0, n_objectives-1)]        

                # SEARCHING SOLUTION-PROCESS INFORMATION
                prev_f_star, prev_feasibility_flag, can_be_skiped_flag = ef_vs_efs_check_vs_f_stars_check(ef, ef_vectors, 
                                                                                                        feasibility_flags, f_stars)

                ef_for_graph = ef.copy()

                if prev_feasibility_flag==3:
                    #print('The subproblem can be skipped as it is a relaxation of infeasible problem')

                    ef = acceleration_algo_with_early_exit(ef, f_min, f_max, n_objectives)
                    f_rwv, ef, termination_flag = main_check_improved(ef, f_min, f_max, f_rwv, n_objectives, jump)
                    run = run+1
                    ef_for_graph.append(99999)
                    for_graph_f_stars.append(f_star_inf)
                    for_graph_feasibility_flag.append(prev_feasibility_flag)

                elif prev_feasibility_flag==2 and can_be_skiped_flag==1:
                    #print('The subproblem can be skipped as it is a relaxation of previously solved problem')
                    ef[0] = prev_f_star[0]
                    f_rwv = acceleration_algo_with_bouncing_steps(prev_f_star, f_rwv, index_main_objective)
                    f_rwv, ef, termination_flag = main_check_improved(ef, f_min, f_max, f_rwv, n_objectives, jump)
                    run = run+1
                    ef_for_graph.append(prev_f_star[-1])
                    for_graph_f_stars.append(np.array([prev_f_star[:][i] for i in correct_order]))
                    for_graph_feasibility_flag.append(prev_feasibility_flag)

                else:
                    ef_vectors.append(np.array(ef[:]))
                    obj_constr.RHS = np.array(ef)
                    main_model.optimize()
                    solution_status_flag = main_model.Status
                    feasibility_flags.append(solution_status_flag)
                    for_graph_feasibility_flag.append(solution_status_flag)

                    if solution_status_flag==3 : #infeasble
                        #print('The problem is infeasible')
                        ef = acceleration_algo_with_early_exit(ef, f_min, f_max, n_objectives)
                        f_stars.append(f_star_inf)
                        ef_for_graph.append(99999)
                        for_graph_f_stars.append(f_star_inf)

                    else:
                        current_solution = rounding_to_integer(main_model.getAttr("X", main_model.getVars()))
                        f_star = objective_matrix @ current_solution
                        ef[0] = f_star[0]
                        f_stars.append(np.array(f_star[:]))
                        x_solutions_train.append(current_solution)

                        for_graph_f_stars.append(np.array([f_star[:][i] for i in correct_order]))
                        ef_for_graph.append(f_star[-1])


                        f_rwv = acceleration_algo_with_bouncing_steps(f_star, f_rwv, index_main_objective)

                    f_rwv, ef, termination_flag = main_check_improved(ef, f_min, f_max, f_rwv, n_objectives, jump)
                    run = run+1

                for_graph_ef_vectors.append(np.array([ef_for_graph[i] for i in correct_order]))
                end_time_run = datetime.datetime.now()
                run_time_run = (end_time_run - start_time_run).total_seconds()
                train_run_times.append(run_time_run)
                train_orders.append(str(test_order))
                
            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).total_seconds()

        # Exclude eps values, objective vectors and feasibility flags for relaxation
        self.for_graph_ef_vectors, self.for_graph_f_stars, self.for_graph_feasibility_flag, temp = relaxation_check(for_graph_f_stars,
                                                                                                               for_graph_ef_vectors, 
                                                                                                               for_graph_feasibility_flag, 
                                                                                                               self.f_columns, 
                                                                                                               self.ef_columns)

        self.x_solutions_train = x_solutions_train
        self.train_run_times = train_run_times
        self.train_orders_list = train_orders
        self.time_train_runs = sum(train_run_times)
        self.n_solutions_phase_1 = temp
        print('Phase I: Pareto optimal soltuions found - ', self.n_solutions_phase_1)
        
    def ml_model_1_best_combo(self):
        """
            Block B
        """
        print('==== ML model 1 training ====')
        start_time = datetime.datetime.now()
        objective_matrix = self.objective_matrix_given.copy()
        objective_matrix = np.array(objective_matrix)

        # ML part #1 - best order

        # Create features - Objective Function Feature 
        maxes, mines = fmin_fmax_estimation(objective_matrix, self.constraint_matrix, self.b_vector, self.n, OutputFlag = 0)
        if np.max(maxes) == 0:
            extremes = np.array(mines)/np.min(mines)
        else:
            extremes = np.array(maxes)/np.max(maxes)        
        obj_matrix_stats = []

        for p in range(0, self.p):
            obj =  objective_matrix[p]
            min = obj.min()
            max = obj.max()

            mean = (obj.mean()-min)/(max-min)
            std = (obj.std()-min)/(max-min)
            q1 = (np.quantile(obj, 0.25)-min)/(max-min)
            q2 = (np.quantile(obj, 0.75)-min)/(max-min) 

            obj_matrix_stats.append([mean, q1, q2, std, extremes[p]])

        df_x = pd.DataFrame([objective_matrix @ x for x in self.x_solutions_train], 
                                 columns = self.f_columns).drop_duplicates()
        
        for col in self.f_columns:
            self.jump_updated_general.append(round(np.quantile(df_x[[col]].sort_values(by=col).diff().dropna(), self.quantile)))

        # Form feature vectors following the order o

        features = []

        for t in range(0, len(self.all_orders)):
            # Objective Function Feature ordered
            temp_order_train = self.all_orders[t]
            temp_obj_matrix_stats = [np.array(obj_matrix_stats[i]) for i in temp_order_train]
            temp_features = list(itertools.chain.from_iterable(temp_obj_matrix_stats))

            # Constraint Space Features
            temp_objective_matrix = [objective_matrix[i].tolist() for i in temp_order_train]
            temp_objective_matrix.pop(-1)
            constraint_space_feature = np.array([abs(prod(x))**(1./(self.p-1)) for x in zip(*temp_objective_matrix)])
            min = constraint_space_feature.min()
            max = constraint_space_feature.max()

            mean = (constraint_space_feature.mean()-min)/(max-min)
            std = (constraint_space_feature.std()-min)/(max-min)
            q1 = (np.quantile(constraint_space_feature, 0.25)-min)/(max-min)
            q2 = (np.quantile(constraint_space_feature, 0.75)-min)/(max-min)

            temp_features.append(mean)
            temp_features.append(std)
            temp_features.append(q1)
            temp_features.append(q2)

            # Infromation about increments - Solution  Space Feature
            temp_incr = [self.jump_updated_general[i]/self.jump_updated_general[temp_order_train[-1]] for i in temp_order_train[:-1]]
            temp_features.extend(temp_incr)
            features.append(temp_features)

        # Prepare train data set
        features_names = ['f'+str(x) for x in range(1,len(features[0])+1)]
        df_all_orders = pd.DataFrame(features, columns=features_names)
        df_all_orders['combo'] = self.all_orders
        df_all_orders['combo'] = df_all_orders['combo'].astype(str)

        df_train = pd.DataFrame(self.train_orders_list, columns=['combo'])
        df_train['time'] = self.train_run_times
        df_train['time'] = (df_train['time']-df_train['time'].min())/(df_train['time'].max()-df_train['time'].min())
        df_train = pd.merge(df_train, df_all_orders, how='left', on='combo')
        y_train = np.array(df_train['time'])
        x_train = np.array(df_train[features_names])

        # Train ML model
        best_params = bayesian_search_main(RandomForestRegressor(), x_train, y_train, task = 'regression')
        regression = RandomForestRegressor(**best_params)
        regression.fit(x_train, y_train)
        y_pred_train = regression.predict(x_train)
        y_pred = regression.predict(np.array(features))

        self.mae = mean_absolute_error(y_train, y_pred_train)
        self.rmse = root_mean_squared_error(y_train, y_pred_train)

        df_all_orders['prediction'] = y_pred
        best_combo_id = df_all_orders['prediction'].idxmin()
        self.best_combo = eval(df_all_orders.loc[best_combo_id, 'combo'])
        self.df_best_combo = df_all_orders[['combo','prediction']].sort_values(by=['prediction'])
        end_time = datetime.datetime.now()
        self.time_ml_model_1 = (end_time - start_time).total_seconds()

    def set_k(self, k):
        """
            Helper function
        """
        self.search_space_regions = k     

    def ml_model_2_clustering(self):
        """
            Block C
        """
        print('==== ML model 2 training ====')
        start_time = datetime.datetime.now()

        # Organize required input data in most optimal order predicted by Block B
        objective_matrix = self.objective_matrix_given.copy()
        objective_matrix = [objective_matrix[i] for i in self.best_combo]
        objective_matrix = np.array(objective_matrix)

        f_max, f_min = fmin_fmax_estimation(objective_matrix, self.constraint_matrix, self.b_vector, self.n)

        if np.max(f_max) == 0:
            self.extremes_clustering = f_min
        else:
            self.extremes_clustering = f_max

        ef_columns = ['ef'+str(x) for x in range(1, self.p+1)]
        train_data_df = pd.DataFrame()

        for i,col in zip(self.best_combo, self.f_columns):
            train_data_df[col] = [x[i] for x in self.for_graph_f_stars]

        for i,col in zip(self.best_combo, ef_columns):
            train_data_df[col] = [x[i] for x in self.for_graph_ef_vectors]

        # Clustering part
        # drop infeasible solutions
        train_data_df = train_data_df.drop(train_data_df[train_data_df['f1']==-1].index)

        df_x = pd.DataFrame([objective_matrix @ x for x in self.x_solutions_train],
                            columns = self.f_columns).drop_duplicates()

        kmeans = KMeans(n_clusters=self.search_space_regions, random_state=0).fit(df_x[self.f_columns])
        df_x['labels'] = kmeans.labels_

        if df_x['labels'].value_counts().min()<self.min_sample_size_per_cluster:
            print('The minimal number of data point per cluster is not reached.')
            print('Consider increasing train iterations number or decreasing k')
            self.clustering_fail_flag = 1
        
        else:
            self.ss = silhouette_score(df_x[self.f_columns], kmeans.labels_)
            self.dbi = davies_bouldin_score(df_x[self.f_columns], kmeans.labels_)

            train_data_df = pd.merge(train_data_df, df_x, how='left')

            self.jumps_updated_regions = {}
            self.search_area_min = {}
            self.search_area_max = {}

            # Compute the step size for each search region (cluster)

            for label in range(0,self.search_space_regions):
                df_temp = df_x[df_x['labels']==label]
                temp_list = []
                temp_list_min = []
                temp_list_max = []
                for col in self.f_columns:
                    temp_list.append(round(np.quantile(df_temp[[col]].sort_values(by=col).diff().dropna(), self.quantile)))
                    temp_list_min.append(int(df_temp[col].min()))
                    temp_list_max.append(int(df_temp[col].max()))
                self.jumps_updated_regions[label] =  temp_list 
                self.search_area_min[label] = temp_list_min 
                self.search_area_max[label] = temp_list_max 

            # Classification part
            # Prepare data set for classification part (scaling)

            train_data_df['f1'] = train_data_df['f1']/self.extremes_clustering[0]

            for col, max_value in zip(ef_columns[1:-1], self.extremes_clustering[1:-1]):
                train_data_df[col] = train_data_df[col]/max_value

            x_train = np.array(train_data_df[ef_columns[1:-1]])
            y_train = train_data_df['labels']
            
            best_params = bayesian_search_main(RandomForestClassifier(), x_train, y_train)
            
            classifier = RandomForestClassifier(**best_params)
            self.classifier = classifier.fit(x_train, y_train)

            y_pred = self.classifier.predict(x_train)
            
            self.accuracy = accuracy_score(y_train, y_pred)
            self.f1_macro = np.mean(f1_score(y_train, y_pred, average='macro'))
            
            train_data_df['pred'] = y_pred
            self.confusion_matrix_ml_2 = pd.pivot_table(train_data_df[['labels', 'pred']], 
                                                        columns='pred', 
                                                        index='labels', aggfunc=len, fill_value=0)
        end_time = datetime.datetime.now()
        self.time_ml_model_2 = (end_time - start_time).total_seconds()

    def numerical_study_1(self):
        """
            Block E for 2phase BC with k >1
        """
        print('==== Numerical study 1 ====')
        # ML, predicted step size  + best combo
        objective_matrix = self.objective_matrix_given.copy()
        objective_matrix = [objective_matrix[i] for i in self.best_combo]
        objective_matrix = np.array(objective_matrix)

        # Select main objective 
        n_decision_var = self.n
        n_objectives = self.p
        index_main_objective = n_objectives - 1
        eps = 1

        other_objective_matrix = np.delete(objective_matrix, index_main_objective, axis=0)
        # Results
        df_results = pd.DataFrame(columns=self.results_columns)
        df_train_info = pd.DataFrame(columns=self.train_info_columns)

        # Initialize loop_control variables
        f_max, f_min = fmin_fmax_estimation(other_objective_matrix, self.constraint_matrix, 
                                            self.b_vector, n_decision_var)

        
        
        f_rwv = f_max.copy()
        ef = f_min.copy()

        # help things
        adjusted_objective_matrix = objective_coefficients_matrix(f_min, f_max, eps, index_main_objective, objective_matrix)
        termination_flag = 0                                        # 1 - end the search
        run = 1

        # Initialize model
        main_model = gp.Model('SAUGMECON')
        main_model.setParam('OutputFlag', 0)
        main_model.setParam('DualReductions', 0)

        # Define decision variables
        x = main_model.addMVar(n_decision_var, vtype=GRB.BINARY, name="X")
        # Set objective
        objective = np.sum([[c @ x] for c in adjusted_objective_matrix])
        main_model.setObjective(objective, GRB.MAXIMIZE)
        # Define constraints
        if self.problem_type == 'knapsack':
            main_model.addConstr(self.constraint_matrix@ x <= self.b_vector)
        elif self.problem_type == 'set_covering':
            main_model.addConstr(self.constraint_matrix@ x >= self.b_vector)
        obj_constr = main_model.addConstr(other_objective_matrix@ x >= np.array(ef))
        main_model.update()

        start_time = datetime.datetime.now()
        solving_run_time = main_model.Runtime
        timeout_start = time.time()

        ef_vectors = [np.array([x[i] for i in self.best_combo[:-1]]) for x in self.for_graph_ef_vectors]
        f_stars = [np.array([x[i] for i in self.best_combo]) for x in self.for_graph_f_stars]
        f_star_inf = np.array([-1]*n_objectives)
        feasibility_flags = self.for_graph_feasibility_flag.copy()
        skipped = []

        jump = [1,1,1,1] 
        
        while termination_flag != 1 and time.time() < timeout_start + self.timeout:# and run<=200000:

            # SEARCHING SOLUTION-PROCESS INFORMATION
            prev_f_star, prev_feasibility_flag, can_be_skiped_flag = ef_vs_efs_check_vs_f_stars_check(ef, ef_vectors, 
                                                                                                    feasibility_flags, f_stars)

            if prev_feasibility_flag==3:
                #print('The subproblem can be skipped as it is a relaxation of infeasible problem')

                skipped.append(1)
                ef = acceleration_algo_with_early_exit(ef, f_min, f_max, n_objectives)
                f_rwv, ef, termination_flag, loop_flag = main_check_loop_flag(ef, f_min, f_max, f_rwv, n_objectives, jump)
                run = run+1

            elif prev_feasibility_flag==2 and can_be_skiped_flag==1:
                #print('The subproblem can be skipped as it is a relaxation of previously solved problem')

                skipped.append(1)

                ef[0] = prev_f_star[0]
                f_rwv = acceleration_algo_with_bouncing_steps(prev_f_star, f_rwv, index_main_objective)
                f_rwv, ef, termination_flag, loop_flag = main_check_loop_flag(ef, f_min, f_max, f_rwv, n_objectives, jump)
                
                run = run+1

            else:
                ef_vectors.append(np.array(ef[:]))

                obj_constr.RHS = np.array(ef)
                main_model.optimize()
                solution_status_flag = main_model.Status
                feasibility_flags.append(solution_status_flag)

                if solution_status_flag==3 : #infeasble
                    #print('The problem is infeasible')
                    ef = acceleration_algo_with_early_exit(ef, f_min, f_max, n_objectives)
                    f_stars.append(f_star_inf)

                else:
                    current_solution = rounding_to_integer(main_model.getAttr("X", main_model.getVars()))
                    f_star = objective_matrix @ current_solution
                    ef[0] = f_star[0]
                    f_stars.append(np.array(f_star[:]))
                    f_rwv = acceleration_algo_with_bouncing_steps(f_star, f_rwv, index_main_objective)

                f_rwv, ef, termination_flag, loop_flag = main_check_loop_flag(ef, f_min, f_max, f_rwv, n_objectives, jump)
                run = run+1
                skipped.append(0)
                solving_run_time += main_model.Runtime

            # The algorithm to predict the region of search space
            if loop_flag==1:
                search_area_predicted = self.classifier.predict([[ef[i]/self.extremes_clustering[i] for i in range(1, n_objectives-1)]])[0]
                ef[0] = self.search_area_min[search_area_predicted][0]
                jump = self.jumps_updated_regions[search_area_predicted]
                loop_flag = 0


        end_time = datetime.datetime.now()
        run_time = (end_time - start_time).total_seconds()

        f_star_true = [x*self.f_star_coef for x in f_stars]
        self.f_stars_check_2 = f_stars

        total_solved, total_infeasible_solutions, unique_pareto_solutions, _, skipped_runs = post_run_perfromance(f_star_true, 
                                                                                                                   feasibility_flags,
                                                                                                                   skipped)

        
        df_results.loc[0, self.results_columns] = (self.p, self.n, str(self.best_combo), (run-1), 
                                                   total_solved, skipped_runs, total_infeasible_solutions, 
                                                   unique_pareto_solutions, run_time, solving_run_time,
                                                   ('BC, clusters '+str(self.search_space_regions))) 
        self.df_results_final = pd.concat([self.df_results_final, df_results])

        df_train_info.loc[0, self.train_info_columns] = (self.p, self.n, str(self.best_combo), self.time_train_runs,
                                                     self.time_ml_model_1, self.time_ml_model_2, self.mae, self.rmse,
                                                     self.ss, self.dbi, self.accuracy, self.f1_macro, 
                                                     ('BC, clusters '+str(self.search_space_regions))) 
        self.df_train_info_final = pd.concat([self.df_train_info_final , df_train_info])
        
    def numerical_study_2(self):
        """
            Block E for 2phase A and 2phase BC with k=1
        """
        print('==== Numerical study 2 ====')
        # No ML, updated step size + best combo
        for test_order, scenario in zip([self.best_combo, self.given_order], ['BC, cluster 1', 'R cluster 1']):
            objective_matrix = self.objective_matrix_given.copy()
            objective_matrix = [objective_matrix[i] for i in test_order]
            objective_matrix = np.array(objective_matrix)

            # Select main objective 
            n_decision_var = self.n
            n_objectives = self.p
            index_main_objective = n_objectives - 1
            eps = 1

            other_objective_matrix = np.delete(objective_matrix, index_main_objective, axis=0)
            # Results
            df_results = pd.DataFrame(columns=self.results_columns)

            # Initialize loop_control variables
            f_max, f_min = fmin_fmax_estimation(other_objective_matrix, self.constraint_matrix, 
                                                self.b_vector, n_decision_var)
            f_rwv = f_max.copy()
            ef = f_min.copy()

            # help things
            adjusted_objective_matrix = objective_coefficients_matrix(f_min, f_max, eps, index_main_objective, objective_matrix)
            termination_flag = 0                                        # 1 - end the search
            run = 1
            solved_run = 1

            # Initialize model
            main_model = gp.Model('SAUGMECON')
            main_model.setParam('OutputFlag', 0)
            main_model.setParam('DualReductions', 0)

            # Define decision variables
            x = main_model.addMVar(n_decision_var, vtype=GRB.BINARY, name="X")
            # Set objective
            objective = np.sum([[c @ x] for c in adjusted_objective_matrix])
            main_model.setObjective(objective, GRB.MAXIMIZE)
            # Define constraints
            if self.problem_type == 'knapsack':
                main_model.addConstr(self.constraint_matrix@ x <= self.b_vector)
            elif self.problem_type == 'set_covering':
                main_model.addConstr(self.constraint_matrix@ x >= self.b_vector)
            obj_constr = main_model.addConstr(other_objective_matrix@ x >= np.array(ef))
            main_model.update()

            start_time = datetime.datetime.now()
            solving_run_time = main_model.Runtime
            timeout_start = time.time()

            ef_vectors = [np.array([x[i] for i in test_order[:-1]]) for x in self.for_graph_ef_vectors]
            f_stars = [np.array([x[i] for i in test_order]) for x in self.for_graph_f_stars]
            feasibility_flags = self.for_graph_feasibility_flag.copy()
            f_star_inf = np.array([-1]*n_objectives)
            skipped = []

            # f_star_inf = np.array([-1]*n_objectives)
            # ef_vectors = [np.array([-1]*(n_objectives-1))]
            # f_stars = [f_star_inf]
            # feasibility_flags = [-1]

            jump = [self.jump_updated_general[i] for i in test_order]
            
            while termination_flag != 1 and time.time() < timeout_start + self.timeout:#run<=200000:#
                # print(ef)


                # SEARCHING SOLUTION-PROCESS INFORMATION
                prev_f_star, prev_feasibility_flag, can_be_skiped_flag = ef_vs_efs_check_vs_f_stars_check(ef, ef_vectors, 
                                                                                                        feasibility_flags, f_stars)

                if prev_feasibility_flag==3:
                    # print('The subproblem can be skipped as it is a relaxation of infeasible problem')

                    skipped.append(1)
                    ef = acceleration_algo_with_early_exit(ef, f_min, f_max, n_objectives)
                    f_rwv, ef, termination_flag, _ = main_check_loop_flag(ef, f_min, f_max, f_rwv, n_objectives, jump)
                    # for_graph_f_stars.append([])

                elif prev_feasibility_flag==2 and can_be_skiped_flag==1:
                    # print('The subproblem can be skipped as it is a relaxation of previously solved problem')

                    skipped.append(1)
                    ef[0] = prev_f_star[0]
                    f_rwv = acceleration_algo_with_bouncing_steps(prev_f_star, f_rwv, index_main_objective)
                    f_rwv, ef, termination_flag, _ = main_check_loop_flag(ef, f_min, f_max, f_rwv, n_objectives, jump)
                else:
                    ef_vectors.append(np.array(ef[:]))

                    obj_constr.RHS = np.array(ef)
                    main_model.optimize()
                    solution_status_flag = main_model.Status
                    feasibility_flags.append(solution_status_flag)

                    if solution_status_flag==3 : #infeasble
                        # print('The problem is infeasible')
                        ef = acceleration_algo_with_early_exit(ef, f_min, f_max, n_objectives)
                        f_stars.append(f_star_inf)
                        # for_graph_f_stars.append([])

                    else:
                        current_solution = rounding_to_integer(main_model.getAttr("X", main_model.getVars()))
                        f_star = objective_matrix @ current_solution
                        ef[0] = f_star[0]
                        # print(f_star)
                        f_stars.append(np.array(f_star[:]))
                        f_rwv = acceleration_algo_with_bouncing_steps(f_star, f_rwv, index_main_objective)

                    f_rwv, ef, termination_flag, _ = main_check_loop_flag(ef, f_min, f_max, f_rwv, n_objectives, jump)
                    solved_run = solved_run+1
                    skipped.append(0)
                    solving_run_time += main_model.Runtime
                run = run+1

            end_time = datetime.datetime.now()
            run_time = (end_time - start_time).total_seconds()

            f_star_true = [x*self.f_star_coef for x in f_stars]

            self.f_stars_check_1 = f_stars

            total_solved, total_infeasible_solutions, unique_pareto_solutions, _, skipped_runs = post_run_perfromance(f_star_true, 
                                                                                                                      feasibility_flags,
                                                                                                                      skipped)
            df_results.loc[0, self.results_columns] = (self.p, self.n, str(test_order), (run-1), 
                                                    total_solved, skipped_runs, total_infeasible_solutions,
                                                    unique_pareto_solutions, run_time, 
                                                    solving_run_time,
                                                    scenario)
            self.df_results_final = pd.concat([self.df_results_final, df_results])

    def full_study(self):
        """
            Runs the Models 2phase A, 2phase BC with k=1 and 2phase BC with k=k
        """
        
        self.file_unpacking()
        self.train_runs()
        self.ml_model_1_best_combo()
        self.ml_model_2_clustering()
        self.numerical_study_1()
        self.numerical_study_2()

        # return self.df_results_final, self.df_train_info_final
    
    def full_study_for_ks(self):
        """
            Runs the Models 2phase A, 2phase BC with k=1 and 2phase BC with k = k_range
        """
        self.file_unpacking()
        self.train_runs()
        self.ml_model_1_best_combo()
        self.numerical_study_2()

        for k in self.k_range:
            self.search_space_regions = k 
            self.ml_model_2_clustering()
            self.numerical_study_1()
        
        # return self.df_results_final, self.df_train_info_final

                
                    