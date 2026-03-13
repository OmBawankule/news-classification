
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def create_features(data_path):

    df = pd.read_csv(data_path)

    X = df["clean_text"]
    y = df["label"]

    vectorizer = TfidfVectorizer(max_features=5000)

    X_vec = vectorizer.fit_transform(X)

    return X_vec, y
