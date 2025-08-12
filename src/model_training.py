import logging
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, text_column="text", target_column="target"):
    """
    Trains a Logistic Regression model using TF-IDF vectorized text.
    """
    try:
        logger.debug("Starting TF-IDF vectorization")
        tfidf = TfidfVectorizer()
        X_train = tfidf.fit_transform(train_df[text_column])
        X_test = tfidf.transform(test_df[text_column])

        y_train = train_df[target_column]
        y_test = test_df[target_column]

        logger.debug("Training Logistic Regression model")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Save model and vectorizer
        os.makedirs("./models", exist_ok=True)
        joblib.dump(model, "./models/spam_classifier.pkl")
        joblib.dump(tfidf, "./models/tfidf_vectorizer.pkl")

        logger.info("Model training completed successfully")
        logger.debug(f"Model saved at ./models/spam_classifier.pkl")
        logger.debug(f"Vectorizer saved at ./models/tfidf_vectorizer.pkl")

        return model, X_test, y_test
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise

def main():
    try:
        logger.debug("Loading train and test datasets")
        train_df = pd.read_csv("./data/raw/train.csv")
        test_df = pd.read_csv("./data/raw/test.csv")

        train_model(train_df, test_df)
    except Exception as e:
        logger.error("Model training pipeline failed: %s", e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
