import pandas as pd
import logging


CATEGORICAL_COLUMNS = ["University_Rating", "Research"]
TARGET_COLUMN = "Admit_Chance"


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, dtype=int)
        return df
    except Exception as e:
        logging.error(f"Encoding failed: {e}")
        raise


def split_features_target(df: pd.DataFrame):
    try:
        X = df.drop([TARGET_COLUMN], axis=1)
        y = df[TARGET_COLUMN]
        return X, y
    except Exception as e:
        logging.error(f"Feature-target split failed: {e}")
        raise