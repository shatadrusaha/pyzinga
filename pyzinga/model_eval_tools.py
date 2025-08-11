"""                     Load the libraries.                     """
import lightgbm as lgbm
import pandas as pd
from . import mlflow_tools as mlft
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
import matplotlib.pyplot as plt
from typing import Dict, Any, Union
import numpy as np
import shap


"""                     Define settings for plotting.                       """
# Default plotting parameters for plots.
DEFAULT_FIGSIZE = (12, 10)
# DEFAULT_HUE = None
# DEFAULT_EDGECOLOR = "black"
# DEFAULT_LEGEND = False
# DEFAULT_STYLE = "darkgrid"
# DEFAULT_PALETTE = "deep"
# DEFAULT_SAVEDIR = None
# DEFAULT_CMAP = "coolwarm"


"""                     User defined functions for model evaluation.                     """
# Function for metric calculation.
def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series], 
    y_pred: Union[np.ndarray, pd.Series],
    params_model_eval: Dict[str, Any],
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for classification or regression problems.

    Parameters
    ----------
    y_true : Union[np.ndarray, pd.Series]
        Ground truth (correct) target values.
    y_pred : Union[np.ndarray, pd.Series]
        Estimated target values, typically predicted by a model.
    params_model_eval : Dict[str, Any]
        Dictionary containing model evaluation parameters.
        Must include 'problem_type' key with value either 'binary_classification' or 'regression'.
        For binary classification, can include 'threshold' (default: 0.5).

    Returns
    -------
    Dict[str, float]
        Dictionary containing calculated metrics appropriate for the problem type.
        
    Raises
    ------
    ValueError
        If problem_type is not one of the supported values.
    """
    # Extract parameters for metrics calculation.
    problem_type = params_model_eval['problem_type']
    threshold = params_model_eval.get('threshold', 0.5)  # Default threshold for binary classification.
    
    # Ensure y_true is a 1D array.
    y_true = y_true.squeeze()

    # Binary classification problem.
    if problem_type == 'binary_classification':
        # Convert predictions to binary using the threshold.
        y_pred_binary = (y_pred >= threshold).astype(int)

        # Calculate binary classification metrics.
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'f1_score': f1_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary),
            'recall': recall_score(y_true, y_pred_binary),
            'roc_auc': roc_auc_score(y_true, y_pred),
            'average_precision': average_precision_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred)
        }
    
    # Regression problem.
    elif problem_type == 'regression':
        # Calculate regression metrics.
        # https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-logarithmic-error
        metrics = {
            'r2_score': r2_score(y_true=y_true, y_pred=y_pred),
            'mae': mean_absolute_error(y_true=y_true, y_pred=y_pred),
            'mse': mean_squared_error(y_true=y_true, y_pred=y_pred),
            'msle': mean_squared_log_error(y_true=y_true, y_pred=y_pred),
            'mape': mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred),
            'rmse': root_mean_squared_error(y_true=y_true, y_pred=y_pred),
            'rmsle': root_mean_squared_log_error(y_true=y_true, y_pred=y_pred),
        }
        pass

    # TODO - Implement multiclass classification metrics.
    # # Multiclass classification problem.
    # elif problem_type == 'multiclass_classification':
    #     print("Multiclass classification metrics are not implemented yet.")

    else:
        raise ValueError("\n\tInvalid problem type. Use 'binary_classification' or 'regression'.\n")

    return metrics

# Function for binary classification model evaluation.
def eval_binary_classification(
    model: lgbm.Booster,
    params_data: Dict[str, Union[pd.DataFrame, np.ndarray, pd.Series]],
    params_model_eval: Dict[str, Any]
) -> None:
    """
    Evaluate a LightGBM binary classification model and save evaluation metrics and plots to MLflow.

    Parameters
    ----------
    model : lgbm.Booster
        Trained LightGBM model to evaluate.
    params_data : Dict[str, Union[pd.DataFrame, np.ndarray, pd.Series]]
        Dictionary containing the data splits used for evaluation.
        Must include keys: 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'.
    params_model_eval : Dict[str, Any]
        Dictionary containing model evaluation parameters.
        Can include 'threshold' (default: 0.5) for binary classification.

    Returns
    -------
    None
        This function logs metrics and artifacts directly to MLflow.
    """
    # Get the classification threshold.
    threshold = params_model_eval.get('threshold', 0.5)

    # Mlflow artifact paths.
    path_artifact_file = "results/files"
    path_artifact_plot = "results/plots"
    
    # Predict on the datasets.
    y_train_preds = model.predict(params_data["X_train"])
    y_train_preds_binary = (y_train_preds >= threshold).astype(int)

    y_val_preds = model.predict(params_data["X_val"])
    y_val_preds_binary = (y_val_preds >= threshold).astype(int)

    y_test_preds = model.predict(params_data["X_test"])
    y_test_preds_binary = (y_test_preds >= threshold).astype(int)

    # Feature importance dataframe.
    df_feature_imp = (
        pd.DataFrame(
            {
                "Feature": model.feature_name(),
                "Importance_split": model.feature_importance(importance_type="split"),
                "Importance_gain": model.feature_importance(
                    importance_type="gain"
                ).round(4),
            }
        )
        .sort_values("Importance_split", ascending=False)
        .reset_index(drop=True)
    )

    mlft.log_artifact(
        artifact=df_feature_imp,
        artifact_name="feature_importance.csv",
        artifact_path=path_artifact_file,
    )

    # Feature importance plots.
    imp_types = [
        "split",
        "gain",
    ]  # Type of importance to plot. Options: 'auto', 'split', 'gain'.
    for imp_type in imp_types:
        plot_feature_imp = lgbm.plot_importance(
            booster=model,
            title=f"Feature Importance Plot - '{imp_type}'",
            importance_type=imp_type,
            max_num_features=20,
            figsize=DEFAULT_FIGSIZE,
        )

        mlft.log_artifact(
            artifact=plot_feature_imp.figure,  # Get the figure from the plot.
            artifact_name=f"feature_importance_{imp_type}.png",
            artifact_path=path_artifact_file,
        )
        plt.close(plot_feature_imp.figure)  # Close the feature importance plot figure.

    # Calculate various metrics.
    dataset_type = ["train", "val", "test"]
    for i, (y_preds, y_preds_binary, y_true, dataset) in enumerate(
        zip(
            [y_train_preds, y_val_preds, y_test_preds],
            [y_train_preds_binary, y_val_preds_binary, y_test_preds_binary],
            [params_data["y_train"], params_data["y_val"], params_data["y_test"]],
            dataset_type,
        )
    ):
        # General metrics.
        metrics_model_eval = {
            f"{dataset}_accuracy": accuracy_score(y_true, y_preds_binary),
            f"{dataset}_f1_score": f1_score(y_true, y_preds_binary),
            f"{dataset}_precision": precision_score(y_true, y_preds_binary),
            f"{dataset}_recall": recall_score(y_true, y_preds_binary),
            f"{dataset}_log_loss": log_loss(y_true, y_preds),
            f"{dataset}_roc_auc": float(roc_auc_score(y_true, y_preds)),
            f"{dataset}_average_precision": float(
                average_precision_score(y_true, y_preds)
            ),
        }
        mlft.log_metrics(metrics=metrics_model_eval)

        # Precision-recall curve.
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)  # Create a custom figure and axes.
        precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_preds)
        pr_auc_score = auc(recall, precision)  # precision-recall AUC score
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot(ax=ax)  # Plot the precision-recall curve on the custom axes.
        ax.set_title(f"Precision-Recall Curve - '{dataset}' (AUC = {pr_auc_score:.2f})")
        mlft.log_artifact(
            artifact=disp.figure_,
            artifact_name=f"precision_recall_curve_{dataset}.png",
            artifact_path=path_artifact_plot,
        )
        plt.close(disp.figure_)  # Close the precision-recall curve figure.

        # ROC-AUC curve.
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)  # Create a custom figure and axes.
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_preds)
        disp = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=float(roc_auc_score(y_true, y_preds))
        )
        disp.plot(ax=ax)  # Plot the ROC curve on the custom axes.
        ax.set_title(
            f"ROC-AUC Curve - '{dataset}' (AUC = {float(roc_auc_score(y_true, y_preds)):.2f})"
        )
        mlft.log_artifact(
            artifact=disp.figure_,
            artifact_name=f"roc_curve_{dataset}.png",
            artifact_path=path_artifact_plot,
        )
        plt.close(disp.figure_)  # Close the ROC curve figure.

        # Confusion matrix.
        _, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)  # Create a custom figure and axes.
        cm = confusion_matrix(y_true=y_true, y_pred=y_preds_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", values_format="d")  # Get the axes object
        ax.set_title(f"Confusion Matrix - '{dataset}' ({threshold = })")
        mlft.log_artifact(
            artifact=disp.figure_,
            artifact_name=f"confusion_matrix_{dataset}.png",
            artifact_path=path_artifact_plot,
        )
        plt.close(disp.figure_)  # Close the confusion matrix figure.

        # Classification report.
        cr = classification_report(
            y_true=y_true, y_pred=y_preds_binary, output_dict=True
        )
        # https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
        cr.update(
            {
                "accuracy": {
                    "precision": None,
                    "recall": None,
                    "f1-score": cr["accuracy"],
                    "support": cr["macro avg"]["support"],
                }
            }
        )
        df_cr = pd.DataFrame(
            cr
        ).transpose()  # Convert the classification report to a DataFrame.
        mlft.log_artifact(
            artifact=df_cr,
            artifact_name=f"classification_report_{dataset}.csv",
            artifact_path=path_artifact_file,
        )

        del metrics_model_eval

# Function for SHAP analysis of a LightGBM model.
def shap_analysis_lgbm(
    model: lgbm.Booster,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    params_model_eval: dict = None
) -> None:
    """
    Perform SHAP analysis on a LightGBM model and log results to MLflow.

    Parameters
    ----------
    model : lgbm.Booster
        Trained LightGBM model.
    X_train : pd.DataFrame
        Training features used for model fitting.
    X_test : pd.DataFrame
        Test features for SHAP value calculation.
    params_model_eval : dict, optional
        Dictionary with SHAP analysis parameters. Can include:
            - 'n_samples': int, number of test samples to use (default: 1000)
            - 'topn_features': int, number of top features to display (default: 20)

    Returns
    -------
    None
        Logs SHAP artifacts and plots to MLflow.
    """
    shap.initjs()

    # Extract model evalauation parameters.
    if params_model_eval is None:
        params_model_eval = {}
    
    n_samples = params_model_eval.get('n_samples', 1000)  # Number of samples for SHAP analysis.
    topn_features = params_model_eval.get('topn_features', 20)

    # Create a TreeExplainer for the LightGBM model.
    explainer = shap.TreeExplainer(model=model, feature_perturbation="tree_path_dependent")

    # Mlflow artifact paths.
    path_artifact_file = 'shap/files'
    path_artifact_plot = 'shap/plots'
    
    # Calculate SHAP values for the specified number of samples from the test set.
    X_test_sample = X_test.sample(n=n_samples, random_state=42) if n_samples < len(X_test) else X_test
    shap_values = explainer(X=X_test_sample)

    # Extract and save the feature importance based on SHAP and the model.
    df_fea_imp = pd.DataFrame(
        data={
            'Feature': X_train.columns,
            'Shap_value_abs_mean': shap_values.abs.mean(axis=0).values,
            # 'Shap_value_abs_max': shap_values.abs.max(axis=0).values,
            'Feature_importance_split': model.feature_importance(importance_type='split'),
            'Feature_importance_gain': model.feature_importance(importance_type='gain'),
        }
    ).sort_values(by='Shap_value_abs_mean', ascending=False).reset_index(drop=True)
    mlft.log_artifact(
        artifact=df_fea_imp,
        artifact_name='feature_importance_shap.csv',
        artifact_path=path_artifact_file
    )

    ### Bar plot of shap values.
    shap_plot_bar = shap.plots.bar(
        shap_values=shap_values,
        max_display=topn_features,
        show=False,  # Do not show the plot immediately.
    )

    # Customize the plot appearance.
    fig = shap_plot_bar.figure # Access the underlying matplotlib figure and axes.
    ax = shap_plot_bar.axes

    fig.set_size_inches(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1])
    ax.set_title("Bar Plot - SHAP Feature Importance", fontsize=16)
    ax.set_xlabel("SHAP value (absolute mean)", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    # plt.show()
    mlft.log_artifact(
        artifact=fig,  # Get the figure from the plot.
        artifact_name='bar_plot.png',
        artifact_path=path_artifact_plot
    )
    plt.close(fig)  # Close the bar plot figure.

    ### Beeswarm plot of shap values.
    shap_plot_beeswarm = shap.plots.beeswarm(
        shap_values=shap_values,
        max_display=topn_features,
        show=False,  # Do not show the plot immediately.
    )

    # Customize the plot appearance.
    fig = shap_plot_beeswarm.figure # Access the underlying matplotlib figure and axes.
    ax = shap_plot_beeswarm.axes

    fig.set_size_inches(DEFAULT_FIGSIZE[0], DEFAULT_FIGSIZE[1])
    ax.set_title("Beeswarm Plot - Impact on Model Output", fontsize=16)
    ax.set_xlabel("SHAP value", fontsize=14)
    ax.set_ylabel("Feature", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    # plt.show()
    mlft.log_artifact(
        artifact=fig,  # Get the figure from the plot.
        artifact_name='beeswarm_plot.png',
        artifact_path=path_artifact_plot
    )
    plt.close(fig)  # Close the beeswarm plot figure.

    ### Scatter plots for top features.
    imp_types = df_fea_imp.columns[1:]  # Get the importance types from the DataFrame.
    
    for type in imp_types:
        # Filter the top features based on the current importance type.
        temp_fe = df_fea_imp[['Feature', type]].sort_values(by=type, ascending=False).reset_index(drop=True)
        temp_fe = temp_fe.head(topn_features).reset_index(drop=True)
        
        # Get the feature names for the top features.
        features = temp_fe['Feature'].tolist()

        # Update the number of top features to plot.
        topn_features = len(features) if len(features) < topn_features else topn_features

        # 2x2 grid scatter plots for top features.
        for i in range(0, topn_features, 4):
            fig, axes = plt.subplots(2, 2, figsize=DEFAULT_FIGSIZE)
            axes = axes.flatten()
            for j in range(4):
                idx = i + j
                if idx < topn_features:
                    feature = features[idx]
                    shap.plots.scatter(
                        shap_values=shap_values[:, feature],
                        show=False,
                        ax=axes[j]
                    )
                    axes[j].axhline(y=0, color='red', linestyle='--')
                    axes[j].set_xlabel(feature, fontsize=10)
                    axes[j].set_ylabel("SHAP value", fontsize=10)
                else:
                    axes[j].axis('off')
            
            # Set a main title for the 2x2 grid.
            fig.suptitle(f"Dependence Scatter Plot (top features - {type.lower().replace('_', ' ')})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            # plt.show()
            # Save the 2x2 grid as a single artifact.
            mlft.log_artifact(
                artifact=fig,
                artifact_name=f"scatter_plot_top_features_{type.lower()}_{i//4+1}.png",
                artifact_path=path_artifact_plot
            )
            plt.close(fig)
