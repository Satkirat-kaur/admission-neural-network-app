import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier


def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=123,
            stratify=y
        )

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=(3,),
            batch_size=50,
            max_iter=500,
            random_state=123,
            activation="tanh"
        )

        model.fit(X_train_scaled, y_train)

        return model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise


def save_artifacts(model, scaler, feature_columns):
    try:
        with open("models/admission_mlp_model.pkl", "wb") as f:
            pickle.dump(model, f)

        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        with open("models/feature_columns.pkl", "wb") as f:
            pickle.dump(feature_columns, f)

        logging.info("Artifacts saved successfully.")

    except Exception as e:
        logging.error(f"Saving artifacts failed: {e}")
        raise