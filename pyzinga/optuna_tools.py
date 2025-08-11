"""                     Load the libraries.                     """
import lightgbm as lgbm
import optuna
import time
import numpy as np
from typing import Dict, Any, Optional
from mlflow.models import infer_signature
from . import mlflow_tools as mlft
from . import model_eval_tools as met

"""                     User defined functions for Optuna.                     """
# Model training function for LightGBM.
def train_lgbm_model(
    params_model: Dict[str, Any],
    params_data: Dict[str, Any],
    params_model_eval: Dict[str, Any],
    best_model: bool = False
) -> Dict[str, float]:
    """
    Train a LightGBM model with given parameters and return evaluation metrics.

    Parameters
    ----------
    params_model : Dict[str, Any]
        Dictionary of LightGBM model parameters.
    params_data : Dict[str, Any]
        Dictionary containing training, validation, and test data.
        Should include 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'.
        Optional 'y_log' boolean flag for log transformation of target (regression only).
    params_model_eval : Dict[str, Any]
        Dictionary of model evaluation parameters.
        Should include 'problem_type' specifying 'regression' or 'binary_classification'.
    best_model : bool, optional
        If True, logs additional evaluation metrics and model analysis.

    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics calculated on the test dataset.

    Raises
    ------
    ValueError
        If log transformation is requested for non-regression problems.
    """
    # Check if target needs to be log transformed. Valid for regression problems ONLY.
    y_log = params_data.get('y_log', False)
    mlft.log_params(params={'y_log': y_log})

    # Unpack the problem type and check compatibility with log transformation.
    problem_type = params_model_eval['problem_type'] # 'regression' or 'binary_classification'
    if problem_type != 'regression' and y_log:
        raise ValueError("Log transformation is only applicable for regression problems.")

    # Unpack the data parameters and log transform the target, if required.
    X_train = params_data['X_train']
    X_val = params_data['X_val']
    X_test = params_data['X_test']
    
    y_train = np.log1p(params_data['y_train']) if y_log else params_data['y_train']
    y_val = np.log1p(params_data['y_val']) if y_log else params_data['y_val']
    # y_test = np.log1p(params_data['y_test']) if y_log else params_data['y_test']

    # Create LightGBM datasets.
    ds_lgbm_train = lgbm.Dataset(data=X_train, label=y_train)
    ds_lgbm_val = lgbm.Dataset(data=X_val, label=y_val, reference=ds_lgbm_train)

    # Log the model parameters to MLflow.
    mlft.log_params(params=params_model)

    # Train the LightGBM model.   
    model_lgbm = lgbm.train(
        params=params_model,
        train_set=ds_lgbm_train,
        valid_sets=[ds_lgbm_val],
    )

    """
    the metrics for the validation set can be obtained from the model object
    print(model_lgbm.best_score)
    """

    # Predict on the test dataset.
    y_test_pred = np.expm1(model_lgbm.predict(data=X_test)) if y_log else model_lgbm.predict(data=X_test)
    
    # Calculate the metrics based on the problem type.
    metrics = met.calculate_metrics(
        y_true=params_data['y_test'], 
        y_pred=y_test_pred, 
        params_model_eval=params_model_eval,
    )
    
    # Get the model signature.
    signature = infer_signature(
        model_input=X_train.iloc[:1], 
        model_output=model_lgbm.predict(data=X_train.iloc[:1])
    )

    # Rename 'metrics' keys and log advanced model evaluation metrics, for best model ONLY.
    if best_model:
        # Rename the metrics for the best model.
        metrics = {f"best_{k}": v for k, v in metrics.items()}

        # Log advanced model evaluation metrics.
        # TODO - add params_model_eval to classification evaluation function.
        if problem_type == 'binary_classification':
            # Log binary classification metrics.
            met.eval_binary_classification(
                model=model_lgbm, 
                params_data=params_data,
                params_model_eval=params_model_eval,  # Pass the model evaluation parameters.
            )
        elif problem_type == 'regression':
            # Log regression metrics.
            met.eval_regression(
                model=model_lgbm, 
                params_data=params_data,
                params_model_eval=params_model_eval,  # Pass the model evaluation parameters.
            )
        else:
            raise ValueError(f"Unknown problem type: '{problem_type}'")
        
        # SHAP analysis.
        met.shap_analysis_lgbm(
            model=model_lgbm,
            X_train=X_train,
            X_test=X_test,
            params_model_eval=params_model_eval,  # Pass the model evaluation parameters.
        )
    
    # Define the path for logging model.
    model_path = "best_model"  if best_model else "model"

    # Log model metrics and model to MLflow.
    mlft.log_metrics(metrics=metrics)
    mlft.log_model(model=model_lgbm, signature=signature, model_art_path=model_path)

    return metrics

# Objective function for Optuna to optimize the LightGBM model.
def objective(
    trial: optuna.Trial, 
    params_lgbm: Dict[str, Dict[str, Any]],
    params_data: Dict[str, Any],
    params_mlflow: Dict[str, Any],
    params_model_eval: Dict[str, Any],
    optimiser_metric: str
) -> float:
    """
    Objective function to optimize the LightGBM model using Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object used for parameter suggestions.
    params_lgbm : Dict[str, Dict[str, Any]]
        Dictionary of LightGBM parameter configurations for optimization.
        Each key is a parameter name, and the value is a dictionary with type and range details.
    params_data : Dict[str, Any]
        Dictionary containing training, validation, and test data.
    params_mlflow : Dict[str, Any]
        Dictionary of MLflow configuration parameters.
    params_model_eval : Dict[str, Any]
        Dictionary of model evaluation parameters.
    optimiser_metric : str
        Name of the metric to optimize.

    Returns
    -------
    float
        The value of the optimization metric for this trial.

    Raises
    ------
    ValueError
        If an unknown parameter type is encountered.
    """
    # Create an empty dictionary to store the model parameters.
    params_model = {}

    # Iterate over the parameters and suggest values based on their types.
    for key, value in params_lgbm.items():
        # Update the parameters for modelling based on its type form user input.
        type_param = value['type']
        # Categorical parameters.
        if type_param == 'categorical':
            dict_temp = {
                key: trial.suggest_categorical(name=key, choices=value['choices'])
            }
            params_model.update(dict_temp)
        # Integer parameters.
        elif type_param == 'int':
            dict_temp = {
                key: trial.suggest_int(name=key, low=value['low'], high=value['high'], step=value['step'], log=value['log'])
            }
            params_model.update(dict_temp)
        # Float parameters.
        elif type_param == 'float':
            dict_temp = {
                key: trial.suggest_float(name=key, low=value['low'], high=value['high'], step=value['step'], log=value['log'])
            }
            params_model.update(dict_temp)
        # Constant parameters.
        elif type_param == 'constant':
            params_model[key] = value['value']
        # Undefined parameter type.
        else:
            raise ValueError(f"Unknown parameter type: '{type_param}'")

    # Store the complete, fixed parameters as trial attributes
    trial.set_user_attr('params_model_full', params_model)

    # Start Mlflow run for the trial.
    with mlft.setup_mlflow(
        parent_run_id=params_mlflow['parent_run_id'],
        run_name=f"trial-{trial.number}",
        nested=True,
    ):
        # Train the LightGBM model with the trial parameters.
        metrics = train_lgbm_model(
            params_model=params_model, 
            params_data=params_data,
            params_model_eval=params_model_eval
        )
    
    return metrics[optimiser_metric]  # Return the metric to optimize (e.g., 'auc').

# Optimization function to run the Optuna study.
def run_optimization(
    params_lgbm: Dict[str, Dict[str, Any]],
    params_data: Dict[str, Any],
    params_mlflow: Dict[str, Dict[str, Any]],
    params_study: Dict[str, Any],
    params_model_eval: Dict[str, Any],
    params_tags: Optional[Dict[str, str]] = None
) -> optuna.Study:
    """
    Run the Optuna optimization study to find optimal model parameters.

    Parameters
    ----------
    params_lgbm : Dict[str, Dict[str, Any]]
        Dictionary of LightGBM parameter configurations for optimization.
        Each parameter should have a dictionary with keys like 'type', 'low', 'high', etc.
    params_data : Dict[str, Any]
        Dictionary containing training, validation, and test data.
    params_mlflow : Dict[str, Dict[str, Any]]
        Dictionary of MLflow configuration parameters including:
            - mlflow_exp_name: Experiment name
            - mlflow_run_name: Run name
            - mlflow_tracking_uri: MLflow tracking server URI
    params_study : Dict[str, Any]
        Dictionary of Optuna study parameters including:
            - direction: Direction of optimization ('minimize' or 'maximize')
            - n_trials: Number of trials to run
            - run_parallel: Boolean flag for parallel execution
            - optimiser_metric: Metric to optimize
    params_model_eval : Dict[str, Any]
        Dictionary of model evaluation parameters.
    params_tags : Optional[Dict[str, str]], optional
        Dictionary of tags to be logged to MLflow.

    Returns
    -------
    optuna.Study
        The completed Optuna study object containing all trials.
    """
    # Create a parent MLflow run for the entire optimization process.
    with mlft.setup_mlflow(
        experiment_name=params_mlflow['mlflow_exp_name'], 
        run_name=params_mlflow['mlflow_run_name'],
        tracking_uri=params_mlflow['mlflow_tracking_uri'],
        nested=False
    ) as parent_run:
        # Log the tags to MLflow.
        if params_tags is not None:
            mlft.log_tags(tags=params_tags)
        
        # Define parameters for parallel or sequential runs.
        n_jobs = -1 if params_study['run_parallel'] else 1  # Use all available CPU cores for parallel execution, or single-threaded execution for simplicity.
        params_mlflow['parent_run_id'] = parent_run.info.run_id if params_study['run_parallel'] else None # Get the parent run ID for nested runs in parallel.

        # Create and run the Optuna study.
        study = optuna.create_study(direction=params_study['direction'])
        study.optimize(
            func=lambda trial: objective(
                trial=trial, 
                params_lgbm=params_lgbm,
                params_data=params_data, 
                params_mlflow=params_mlflow,
                params_model_eval=params_model_eval,
                optimiser_metric=params_study['optimiser_metric'],
            ),
            n_trials=params_study['n_trials'],
            timeout=None,
            n_jobs=n_jobs,
            catch=(Exception,),  # Catch all exceptions during the optimization.
            gc_after_trial=True, # Enable garbage collection after each trial to free up memory.
            show_progress_bar=True, # Show a progress bar for the optimization process.
        )

        # Check all the trials have finished or not, when running in parallel.
        if params_study['run_parallel']:
            # Get the number of finished trials.
            trial_count_finished = sum(
                optuna.trial.TrialState.is_finished(t.state) for t in study.trials
            )

            # Wait until all trials are finished.
            while trial_count_finished < params_study['n_trials']:
                print(f"Waiting for {(params_study['n_trials'] - trial_count_finished)} trials to finish...")
                
                trial_count_finished = sum(
                    optuna.trial.TrialState.is_finished(t.state) for t in study.trials
                )
                # Wait for a short period before checking again.
                time.sleep(30)
        
        # Log the Optuna study results to MLflow.
        flag_study = mlft.log_optuna_study(study=study, params_study=params_study)

        # If the study was logged successfully, log the best model.
        if flag_study:
            print("Logging the best model to MLflow...\n")
            
            # Get the model parameters for the best trial, using the 'user_attrs'.
            params_model_best = study.best_trial.user_attrs['params_model_full']
            
            # Build the best model.
            _ = train_lgbm_model(
                params_model=params_model_best, 
                params_data=params_data,
                params_model_eval=params_model_eval,  # Pass the model evaluation parameters.
                best_model=True,  # Log advanced model metrics for the best model.
            )
            
    return study
