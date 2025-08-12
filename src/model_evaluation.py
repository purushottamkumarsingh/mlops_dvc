import logging
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def evaluate_model(model_path="./models/spam_classifier.pkl",
                   vectorizer_path="./models/tfidf_vectorizer.pkl",
                   test_data_path="./data/raw/test.csv",
                   text_column="text", target_column="target"):
    """
    Evaluates the trained model on the test dataset.
    """
    try:
        logger.debug("Loading trained model and vectorizer")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        logger.debug("Loading test data")
        test_df = pd.read_csv(test_data_path)

        logger.debug("Vectorizing test text")
        X_test = vectorizer.transform(test_df[text_column])
        y_test = test_df[target_column]

        logger.debug("Making predictions")
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="spam")
        rec = recall_score(y_test, y_pred, pos_label="spam")
        f1 = f1_score(y_test, y_pred, pos_label="spam")
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall: {rec:.4f}")
        logger.info(f"F1-score: {f1:.4f}")
        logger.debug(f"Confusion Matrix:\n{cm}")
        logger.debug(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise

def main():
    try:
        evaluate_model()
    except Exception as e:
        logger.error("Model evaluation pipeline failed: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
