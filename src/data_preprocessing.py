import pandas as pd
import logging


def load_data(path: str) -> pd.DataFrame:
    try:
        logging.info(f"Loading dataset from {path}")
        return pd.read_csv(path)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        if "Serial_No" in df.columns:
            df = df.drop(["Serial_No"], axis=1)

        df["University_Rating"] = df["University_Rating"].astype("object")
        df["Research"] = df["Research"].astype("object")

        df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)

        return df

    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise