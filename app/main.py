from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')

from typing import Literal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logged_model = "mlartifacts/444505746376893658/55245fcf8a8b465abf3ee4a12e556f4a/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)
logger.info(f"Loading model")

app = FastAPI(title="Churn Prediction API")

import pandas as pd
import joblib
import os

TRANSFORMER_PATH = "mlartifacts/transformer.pkl"
def load_transformer(path=TRANSFORMER_PATH):
    """
    Load the saved ColumnTransformer.

    Args:
        path (str): Path to the transformer.pkl file

    Returns:
        ColumnTransformer: Trained transformer
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transformer file not found at: {path}")
    transformer = joblib.load(path)
    return transformer

def preprocess_input(input_data: dict, transformer):
    """
    Preprocess input data for prediction.

    Args:
        input_data (dict): Raw input dictionary from API
        transformer (ColumnTransformer): Fitted transformer

    Returns:
        pd.DataFrame: Preprocessed data ready for model prediction
    """
    df = pd.DataFrame([input_data])  # Convert input dict to single-row DataFrame
    transformed = transformer.transform(df)
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
    return transformed_df

TRANSFORMER = load_transformer()

class ChurnData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: Literal[0, 1]
    IsActiveMember: Literal[0, 1] 
    EstimatedSalary: float

@app.get("/")
def home():
    logger.info("Home endpoint called")
    return {"message": "Welcome to the Churn Prediction API!"}

@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: ChurnData):
    try:
        logger.info(f"Prediction requested with input: {data}")

        input_df = preprocess_input(dict(data), TRANSFORMER)
        prediction = model.predict(input_df)
        result = int(prediction[0])

        logger.info(f"Prediction result: {result}")
        return {"churn_prediction": result}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return HTTPException(status_code=500, detail=e)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)