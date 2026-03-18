import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

def pre_processing(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["MAL_ID","Name_x","Genres_x","sypnopsis","Score_y"]]
    
    df["Tag"] = df["Genres_x"] +" "+ df["sypnopsis"]
    df = df.drop(columns=["Genres_x","sypnopsis"])
    df = df.dropna()
    return df

def save_data(df: pd.DataFrame, path: str):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the dataframe
    df.to_csv(path, index=False)
    print("Data saved successfully!")

def main():
    data_path1 = r"./experiments/anime.csv"
    data_path2 = r"./experiments/anime_with_synopsis.csv"
    anime = load_data(data_path1)
    sys = load_data(data_path2)
    df = anime.merge(sys,on="MAL_ID")
    df = pre_processing(df)
    df_path = r".\data\raw\data.csv"
    save_data(df, df_path)

if __name__ == "__main__":
    main()