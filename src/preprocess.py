import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(path: str)-> pd.DataFrame:
    df=pd.read_csv(path)
    if "target" not in df.columns and "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
    return df

def get_feature_lists(df: pd.DataFrame):
    numeric_features = [col for col in ["age", "trestbps", "chol", "thalch", "oldpeak"] if col in df.columns]
    categorical_features = [col for col in ["sex", "cp", "fbs", "restecg", "exang"] if col in df.columns]
    return numeric_features, categorical_features

def build_preprocessor(numeric_features, categorical_features):
    preprocessor= ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor