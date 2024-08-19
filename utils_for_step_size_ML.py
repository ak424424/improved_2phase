from hyperopt import fmin, tpe, STATUS_OK, Trials, space_eval, hp
import numpy as np
from functools import partial
from sklearn.model_selection import StratifiedKFold, cross_validate, KFold
from sklearn.base import clone

# 
cv = 5
n_iterations = 50
n_iterations_ml1 = 2

# Random forest - hyperparameters range
rf_n_estimators_min = 2
rf_n_estimators_max = 50
rf_n_estimators_step = 5
rf_max_depth_min = 3
rf_max_depth_max = 20
rf_max_depth_step = 2
rf_min_samples_leaf = [5, 10, 20]
rf_max_features = ['log2', 'sqrt']
rf_bootstrap = [True, False]
rf_criterion = ['gini', 'entropy']
rfc_criterion = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
rf_class_weight = ['balanced', 'balanced_subsample']


def rf_bayes_param_grid_generator(n_estimators_param_min=rf_n_estimators_min,
                                  n_estimators_param_max=rf_n_estimators_max,
                                  max_depth_param_min=rf_max_depth_min, max_depth_param_max=rf_max_depth_max,
                                  min_samples_leaf_param=rf_min_samples_leaf,
                                  max_features_param=rf_max_features,
                                  bootstrap_param=rf_bootstrap,
                                  criterion_param=rf_criterion,
                                  class_weight_param=rf_class_weight):
    """
    Creates the hyperparameters search space for Random Forest optimization through Bayesian Optimization.

        Args:
          n_estimators_param_min: An integer, a minimal value for n_estimators parameter
          n_estimators_param_max: An integer, a maximal value for n_estimators parameter
          max_depth_param_min: An integer, a minimal value for max_depth parameter
          max_depth_param_max: An integer, a maximal value for max_depth parameter
          min_samples_leaf_param: A dictionary with values for min_samples_leaf parameter
          max_features_param: A dictionary with values for max_features parameter
          bootstrap_param: A dictionary with values for bootstrap parameter
          criterion_param: A dictionary with values for criterion parameter
          class_weight_param: A dictionary with values for class_weight parameter

        Returns:
          The hyperparameters search space for Random Forest optimization through Bayesian Optimization
    """


    space = {'criterion': hp.choice('criterion', criterion_param),
             'max_depth': hp.choice('max_depth', np.arange(max_depth_param_min,
                                                           max_depth_param_max + 1, dtype=int)),
             'max_features': hp.choice('max_features', max_features_param),
             'min_samples_leaf': hp.choice('min_samples_leaf', min_samples_leaf_param),
             'n_estimators': hp.choice('n_estimators', np.arange(n_estimators_param_min,
                                                                 n_estimators_param_max + 1, dtype=int)),
             'bootstrap': hp.choice('bootstrap', bootstrap_param),
             'class_weight': hp.choice('class_weight', class_weight_param)
             }

    return space

def rfc_bayes_param_grid_generator(n_estimators_param_min=rf_n_estimators_min,
                                  n_estimators_param_max=rf_n_estimators_max,
                                  max_depth_param_min=rf_max_depth_min, max_depth_param_max=rf_max_depth_max,
                                  min_samples_leaf_param=rf_min_samples_leaf,
                                  max_features_param=rf_max_features,
                                  bootstrap_param=rf_bootstrap,
                                  criterion_param=rfc_criterion):
    """
    Creates the hyperparameters search space for Random Forest optimization through Bayesian Optimization.

        Args:
          n_estimators_param_min: An integer, a minimal value for n_estimators parameter
          n_estimators_param_max: An integer, a maximal value for n_estimators parameter
          max_depth_param_min: An integer, a minimal value for max_depth parameter
          max_depth_param_max: An integer, a maximal value for max_depth parameter
          min_samples_leaf_param: A dictionary with values for min_samples_leaf parameter
          max_features_param: A dictionary with values for max_features parameter
          bootstrap_param: A dictionary with values for bootstrap parameter
          criterion_param: A dictionary with values for criterion parameter
          class_weight_param: A dictionary with values for class_weight parameter

        Returns:
          The hyperparameters search space for Random Forest optimization through Bayesian Optimization
    """


    space = {'criterion': hp.choice('criterion', criterion_param),
             'max_depth': hp.choice('max_depth', np.arange(max_depth_param_min,
                                                           max_depth_param_max + 1, dtype=int)),
             'max_features': hp.choice('max_features', max_features_param),
             'min_samples_leaf': hp.choice('min_samples_leaf', min_samples_leaf_param),
             'n_estimators': hp.choice('n_estimators', np.arange(n_estimators_param_min,
                                                                 n_estimators_param_max + 1, dtype=int)),
             'bootstrap': hp.choice('bootstrap', bootstrap_param)
             }

    return space

def bayesian_search_objective(params, clf, x, y, scoring_name, cv_method):
    """
    The support function for Bayesian Optimization.
    Defines a classification model for further Bayesian Optimization.

        Args:
          params: A dictionary, the custom set of hyperparameters
          clf: An estimator object, a raw model to be optimized.
          x: An array or dataframe, the explanatory variables
          y: An array or dataframe, a response variable

        Returns:
          The model's loss and parameters.
    """
    clf.set_params(**params)
    skf = cv_method(n_splits=cv, shuffle=True, random_state=42)
    score_summary = cross_validate(estimator=clf, X=x, y=y,
                                    scoring = scoring_name,
                                    cv=skf,
                                    return_train_score=True)

    train_score = np.mean(np.array(score_summary['train_score']))
    test_score = np.mean(np.array(score_summary['test_score']))
    return {'loss': - test_score,
            'train_score': - train_score,
            'test_score': - test_score,
            'params': params,
            'status': STATUS_OK}

def bayesian_search_main(clf, x, y, task='classification'):
    """
    The main function for Bayesian Optimization.
    Performs model's hyperparameters tuning through Bayesian Search CV.

        Args:
          x: An array or dataframe, the explanatory variables
          y: An array or dataframe, a response variable

        Returns:
          The best parameters and score.
    """
    if task == 'classification':
      scoring_metrcics_name = 'f1_macro'
      n_iterations_used = n_iterations
      grid = rf_bayes_param_grid_generator()
      cv_method = StratifiedKFold
    
    elif task == 'regression':
      scoring_metrcics_name = 'neg_mean_squared_error'
      n_iterations_used = n_iterations_ml1
      grid = rfc_bayes_param_grid_generator()
      cv_method = KFold
    
    else:
      print('Unknown task')

    clf = clone(clf)
    obj = partial(bayesian_search_objective, clf=clf, x=x, y=y, scoring_name=scoring_metrcics_name, cv_method=cv_method)
    trials = Trials()


    best = fmin(
        fn=obj,
        space=grid,
        algo=tpe.suggest,  # Bayesian Optimization
        max_evals=n_iterations_used,
        trials=trials,
        rstate=np.random.default_rng(seed=42)
    )

    #test_score = -min(trials.losses())
    best_params = space_eval(grid, best)


    return best_params#, test_score, trials.results

