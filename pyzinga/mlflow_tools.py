import os
import tempfile
from typing import Dict, Any, Optional
from datetime import timedelta
import mlflow
import optuna
import optuna.visualization as vis
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pathlib
import joblib


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    nested: bool = False,
) -> mlflow.ActiveRun:
    """Set up MLflow experiment and create a new run."""
    # Validate inputs.
    # if not tracking_uri and not experiment_name and (not run_id or not run_name or not parent_run_id):
    #     raise ValueError(
    #         "At least one of the following must be provided:\n\t'tracking_uri', 'experiment_name', or 'run_id/run_name/parent_run_id'"
    #     )      

    # Set tracking URI, if provided,
    if tracking_uri:
        mlflow.set_tracking_uri(uri=tracking_uri)
    
    # Set experiment name, if provided.
    if experiment_name:
        mlflow.set_experiment(experiment_name=experiment_name)

    return mlflow.start_run(
        run_id=run_id, run_name=run_name, nested=nested, parent_run_id=parent_run_id
    )


def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    for param_name, param_value in params.items():
        mlflow.log_param(key=param_name, value=param_value)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to MLflow."""
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(key=metric_name, value=metric_value, step=step)


def log_model(
    model: Any, signature, model_art_path: str = "model", conda_env: Optional[Dict] = None
) -> None:
    """Log a model to MLflow."""
    mlflow.lightgbm.log_model(
        lgb_model=model, signature=signature, name=model_art_path, conda_env=conda_env
    )


def log_artifact(
    artifact, 
    artifact_name: str,
    artifact_path: str
) -> None:
    """
    Save various types of artifacts including figures, dataframes, and other data objects.
    Supports both matplotlib figures (saved as PNG) and plotly figures (saved as HTML).
    """
    # Get file extension
    file_extension = pathlib.Path(artifact_name).suffix.lower()

    # Handle based on artifact type or file extension
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create full path for the artifact locally.
        filepath = os.path.join(tmpdirname, artifact_name)

        # For matplotlib figures
        if isinstance(artifact, plt.Figure):
            artifact.savefig(fname=filepath, bbox_inches='tight', dpi=300)
            
        # For plotly figures
        elif isinstance(artifact, go.Figure):
            artifact.write_html(filepath)
            
        # For pandas DataFrames
        elif isinstance(artifact, pd.DataFrame):
            if file_extension == '.csv':
                artifact.to_csv(filepath, index=False)
            elif file_extension == '.parquet':
                artifact.to_parquet(filepath, index=False)
            elif file_extension in ['.pkl', '.pickle']:
                joblib.dump(value=artifact, filename=filepath)  # Use joblib for .pkl and .pickle
            else:
                raise ValueError(f"Unsupported file extension '{file_extension}' for DataFrame. Use .csv, .parquet, .pkl, or .pickle")
        
        # For other file types based on extension
        else:
            if file_extension in ['.pkl', '.pickle']:
                joblib.dump(value=artifact, filename=filepath)
            elif file_extension == '.txt':
                with open(filepath, 'w') as f:
                    f.write(str(artifact))
            else:
                raise ValueError(f"Unsupported artifact type or file extension: {file_extension}")

        mlflow.log_artifact(local_path=filepath, artifact_path=artifact_path)


def log_optuna_study(study, params_study) -> bool:
    """Log Optuna study details to MLflow."""
    # Log study attributes, best metric value.
    log_params(params={f"study_{k}": v for k, v in params_study.items()})
    log_metrics(metrics={"best_value": study.best_value})

    # Get the number of trials that completed successfully.
    trial_count_complete = sum(
        [1 if t.state == optuna.trial.TrialState.COMPLETE else 0 for t in study.trials]
    )

    # Log the study results to MLflow, ONLY if all trials finished successfully.
    if trial_count_complete == params_study['n_trials']:
        print(
            f"\nAll {trial_count_complete} trials completed successfully.\nLogging the study results to MLflow...\n"
        )

        #  Get the study details in a DataFrame and log it as an artifact.
        df_study = pd.DataFrame(
            data={
                "number": [trial.number for trial in study.trials],
                "state": [trial.state.name for trial in study.trials],
                "value": [trial.value for trial in study.trials],
                "datatime_start": [trial.datetime_start.isoformat() if trial.datetime_start else None for trial in study.trials],
                "datetime_complete": [trial.datetime_complete.isoformat() if trial.datetime_complete else None for trial in study.trials],
                "duration_hms": [
                    str(timedelta(seconds=trial.duration.total_seconds())).split(sep=".")[0] 
                    if trial.duration else None for trial in study.trials
                ],
                "params": [trial.params for trial in study.trials],
            }
        )
        log_artifact(artifact=df_study, artifact_name="study_details.csv", artifact_path="optuna_study")

        # Log optimisation history, parameter importances, and parallel coordinate plots.
        fig = vis.plot_optimization_history(study=study)
        log_artifact(artifact=fig, artifact_name="optimization_history.html", artifact_path="optuna_plots")
        
        fig = vis.plot_param_importances(study=study)
        log_artifact(artifact=fig, artifact_name="param_importances.html", artifact_path="optuna_plots")
        
        fig = vis.plot_parallel_coordinate(study=study)
        log_artifact(artifact=fig, artifact_name="parallel_coordinate.html", artifact_path="optuna_plots")

        return True
        
    else:
        # Don't log the study results, if not all trials completed successfully.
        print(
            f"\nWarning:\n\tOnly {trial_count_complete} out of {params_study['n_trials']} trials completed successfully.\n\tSome trials may have failed or been interrupted.\n\tSkipping the MLflow logging of the study results."
        )
        return False
