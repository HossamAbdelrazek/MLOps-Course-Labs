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