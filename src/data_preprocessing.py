import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
import os

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df["Score_y"] = df["Score_y"].replace({
    "Unknown":np.nan
    })
    df["Score_y"] = pd.to_numeric(df["Score_y"], errors="coerce")
    df["Score_y"] = df["Score_y"].fillna(df["Score_y"].median())
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    l = []
    for text in df["Tag"]:
       text = text.lower()
       text = ''.join(char for char in text if char.isalnum() or char.isspace())
       words = text.split()
       words = [stemmer.stem(word) for word in words if word not in stop_words]
       l.append(' '.join(words))

    df["Tag"] = l
    return df


def save_data(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def main():
    df_path = r"./data/raw/data.csv"
    save_df_path = r"./data/processed/data_processed.csv"
    df = load_data(df_path)

    df_processed = preprocessing(df)

    save_data(df_processed, save_df_path)

if __name__ == "__main__":
    main()