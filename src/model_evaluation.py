import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test_scaled, y_test):
    try:
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)
        return accuracy, cm, report
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise