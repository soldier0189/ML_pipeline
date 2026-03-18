import pandas as pd
import numpy as np
import pickle
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

with open("params.yaml") as f:
    params = yaml.safe_load(f)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def feature_engineering(df: pd.DataFrame):
    df["Tag"] = df["Tag"].fillna("")
    tfidf = TfidfVectorizer(stop_words='english', max_features= params["tfidf"]["max_features"])
    tfidf_matrix = tfidf.fit_transform(df['Tag'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity, tfidf

def save_model(model, path: str):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    with open(path, "wb") as f:
        pickle.dump(model, f)

def main():
    df_path = r"./data/processed/data_processed.csv"
    df = load_data(df_path)
    similarity, tfidf = feature_engineering(df)
    save_model(similarity, "./artifacts/similarity.pkl")
    save_model(tfidf, "./artifacts/tfidf.pkl")
        

if __name__ == "__main__":
    main()
