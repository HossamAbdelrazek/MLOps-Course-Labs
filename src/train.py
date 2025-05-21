"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
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


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)

    # Log model

    ### Log the data

    return log_reg


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("https://localhost:5000")
    ### Set the experiment name
    mlflow.set_experiment("Experiment_1")

    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run():
        ### Log the run ID
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        df = pd.read_csv("data/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        log_artifact("transformer.pkl", col_transf)

        ### Log the max_iter parameter

        model = train(X_train, y_train)

        
        y_pred = model.predict(X_test)

        ### Log metrics after calculating them


        ### Log tag


        
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        
        # Log the image as an artifact in MLflow
        
        plt.show()


if __name__ == "__main__":
    main()
