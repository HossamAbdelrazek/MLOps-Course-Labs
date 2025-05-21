"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

from mlflow.models.signature import infer_signature
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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
        model: trained model
    """
    model = lgb(n_estimators=200,max_depth=5,learning_rate=0.01)
    model.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = infer_signature(X_train, y_train)
    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="LightGBM",
        signature=signature
    )
    ### Log the data
    mlflow.log_artifact('dataset/Churn_Modelling.csv', artifact_path="Data")
    return model


def main():
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    ### Set the experiment name
    mlflow.set_experiment("Experiment_1")

    ### Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run():

        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### Log the parameters
        model = train(X_train, y_train)
        mlflow.log_params(model.get_params())

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
        mlflow.set_tag("model_type", "LightGBM Classifier")

        
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)

        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        plot_path = "plots/confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()
        # Log the image as an artifact in MLflow
        
        mlflow.log_artifact(plot_path, artifact_path="confusion_matrix")

if __name__ == "__main__":
    main()
