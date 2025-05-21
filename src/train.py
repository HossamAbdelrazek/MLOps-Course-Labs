"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import mlflow.models
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from preprocessing import preprocess
import mlflow


def train(X_train, y_train, max_iter=1000):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=max_iter)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = mlflow.models.infer_signature(X_train, y_train)
    # Log model
    mlflow.sklearn.log_model(
        log_reg,
        artifact_path="model",
        registered_model_name="LogisticRegressionModel",
        signature=signature
    )
    ### Log the data
    data = pd.concat([X_train, y_train], axis=1)
    mlflow.log_artifact(data, artifact_path="training_data")

    return log_reg


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    ### Set the experiment name
    mlflow.set_experiment("Experiment_1")

    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run():

        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the max_iter parameter
        max_iter = 1000
        model = train(X_train, y_train, max_iter)
        mlflow.log_param("max_iter",max_iter)

        y_pred = model.predict(X_test)

        ### Log metrics after calculating them
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)

        ### Log tag
        mlflow.set_tag("model_type", "Logistic Regression")

        
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)

        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        
        # Log the image as an artifact in MLflow
        plt.show()
        plt.savefig("plots/confusion_matrix.png")
        mlflow.log_artifact("plots/confusion_matrix.png", artifact_path="confusion_matrix")

if __name__ == "__main__":
    main()
