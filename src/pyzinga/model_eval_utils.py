"""                     Import libraries.                       """
import lightgbm as lgbm
import pandas as pd
import utils.mlflow_utils as mlfu
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, log_loss, auc, 
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
)
import matplotlib.pyplot as plt

# Function to evaluate a LightGBM model and log metrics and plots to MLflow.
def model_eval_binary_classification(
    model: lgbm.Booster, 
    params_data: dict,
    threshold: float = 0.5,
) -> None:
    """Evaluate a LightGBM model and save evaluation metrics and plots to mlflow."""

    # Mlflow artifact paths.
    path_artifact_azure_file = 'results/files'
    path_artifact_azure_plot = 'results/plots'
    # path_artifact_azure_model = 'model'

    # Predict on the datasets.
    y_train_preds = model.predict(params_data['X_train'])
    y_train_preds_binary = (y_train_preds >= threshold).astype(int)
    
    y_val_preds = model.predict(params_data['X_val'])
    y_val_preds_binary = (y_val_preds >= threshold).astype(int)

    y_test_preds = model.predict(params_data['X_test'])
    y_test_preds_binary = (y_test_preds >= threshold).astype(int)

    # Feature importance dataframe.
    df_feature_imp = pd.DataFrame({
            'Feature': model.feature_name() ,
            'Importance_split': model.feature_importance(importance_type='split'),
            'Importance_gain': model.feature_importance(importance_type='gain').round(4),
        }).sort_values('Importance_split', ascending=False).reset_index(drop=True)
    
    mlfu.log_artifact(
        artifact=df_feature_imp,
        artifact_name='feature_importance.csv',
        artifact_path=path_artifact_azure_file
        )
    
    # Feature importance plots.
    imp_types = ['split', 'gain']  # Type of importance to plot. Options: 'auto', 'split', 'gain'.
    for imp_type in imp_types:
        plot_feature_imp = lgbm.plot_importance(
            booster=model, 
            title=f"Feature Importance Plot - '{imp_type}'",
            importance_type=imp_type,
            max_num_features=20, 
            figsize=(10, 8)
        )

        mlfu.log_artifact(
            artifact=plot_feature_imp.figure,  # Get the figure from the plot.
            artifact_name=f"feature_importance_{imp_type}.png",
            artifact_path=path_artifact_azure_file
            )

    # Calculate various metrics.
    dataset_type = ['train', 'val', 'test']
    for i, (y_preds, y_preds_binary, y_true, dataset) in enumerate(
        zip(
            [y_train_preds, y_val_preds, y_test_preds],
            [y_train_preds_binary, y_val_preds_binary, y_test_preds_binary],
            [params_data['y_train'], params_data['y_val'], params_data['y_test']],
            dataset_type
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
            f"{dataset}_average_precision": float(average_precision_score(y_true, y_preds)),
        }
        mlfu.log_metrics(metrics=metrics_model_eval)
        
        # Precision-recall curve.
        _, ax = plt.subplots(figsize=(10, 8)) # Create a custom figure and axes. width=10, height=8
        precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_preds)
        pr_auc_score = auc(recall, precision) # precision-recall AUC score
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot(ax=ax)  # Plot the precision-recall curve on the custom axes.
        ax.set_title(f"Precision-Recall Curve - '{dataset}' (AUC = {pr_auc_score:.2f})")
        mlfu.log_artifact(
            artifact=disp.figure_,
            artifact_name=f"precision_recall_curve_{dataset}.png",
            artifact_path=path_artifact_azure_plot
        )
        
        # ROC-AUC curve.
        _, ax = plt.subplots(figsize=(10, 8)) # Create a custom figure and axes. width=10, height=8
        fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_preds)
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=float(roc_auc_score(y_true, y_preds)))
        disp.plot(ax=ax)  # Plot the ROC curve on the custom axes.
        ax.set_title(f"ROC-AUC Curve - '{dataset}' (AUC = {float(roc_auc_score(y_true, y_preds)):.2f})")
        mlfu.log_artifact(
            artifact=disp.figure_,
            artifact_name=f"roc_curve_{dataset}.png",
            artifact_path=path_artifact_azure_plot
        )

        # Confusion matrix.
        _, ax = plt.subplots(figsize=(10, 8)) # Create a custom figure and axes. width=10, height=8
        cm = confusion_matrix(y_true=y_true, y_pred=y_preds_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', values_format='d')  # Get the axes object
        ax.set_title(f"Confusion Matrix - '{dataset}' ({threshold = })")
        mlfu.log_artifact(
            artifact=disp.figure_,
            artifact_name=f"confusion_matrix_{dataset}.png",
            artifact_path=path_artifact_azure_plot
        )

        # Classification report.
        cr = classification_report(y_true=y_true, y_pred=y_preds_binary, output_dict=True)
        # https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format
        cr.update({"accuracy": {"precision": None, "recall": None, "f1-score": cr["accuracy"], "support": cr['macro avg']['support']}})
        df_cr = pd.DataFrame(cr).transpose()  # Convert the classification report to a DataFrame.
        mlfu.log_artifact(
            artifact=df_cr,
            artifact_name=f"classification_report_{dataset}.csv",
            artifact_path=path_artifact_azure_file
        )
        
        del metrics_model_eval
