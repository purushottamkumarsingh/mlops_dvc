import logging
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def vectorize_text(train_df: pd.DataFrame, test_df: pd.DataFrame, text_column: str = "text"):
    """
    Vectorizes the text column using TF-IDF and returns transformed datasets.
    """
    try:
        logger.debug("Starting TF-IDF vectorization")
        tfidf = TfidfVectorizer()
        X_train = tfidf.fit_transform(train_df[text_column])
        X_test = tfidf.transform(test_df[text_column])
        logger.debug("TF-IDF vectorization completed")
        return X_train, X_test, tfidf
    except Exception as e:
        logger.error("Error during vectorization: %s", e)
        raise

def save_features(X_train, X_test, y_train, y_test, vectorizer, output_dir="./data/processed"):
    """
    Saves processed features and labels to disk.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(X_train, os.path.join(output_dir, "X_train.pkl"))
        joblib.dump(X_test, os.path.join(output_dir, "X_test.pkl"))
        joblib.dump(y_train, os.path.join(output_dir, "y_train.pkl"))
        joblib.dump(y_test, os.path.join(output_dir, "y_test.pkl"))
        joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
        logger.debug(f"Processed features saved in {output_dir}")
    except Exception as e:
        logger.error("Error while saving features: %s", e)
        raise

def main():
    try:
        # If you have preprocessing.py output:
        # train_df = pd.read_csv("./data/processed/train_preprocessed.csv")
        # test_df = pd.read_csv("./data/processed/test_preprocessed.csv")

        # Or directly from raw ingestion step:
        train_df = pd.read_csv("./data/raw/train.csv")
        test_df = pd.read_csv("./data/raw/test.csv")

        X_train, X_test, tfidf = vectorize_text(train_df, test_df, text_column="text")
        y_train, y_test = train_df["target"], test_df["target"]

        save_features(X_train, X_test, y_train, y_test, tfidf)
        logger.info("Feature engineering pipeline completed successfully")
    except Exception as e:
        logger.error("Feature engineering pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()
