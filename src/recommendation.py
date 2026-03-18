import pandas as pd
import pickle
import os

def load_data(path: str):
    # Load based on file type
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file format")

def recommend(anime, similarity, anime_name, top_n=5):
    # Find the index of the anime
   
    idx = anime[anime['Name_x'] == anime_name].index[0]
    
    # Get similarity scores for this anime with all others
    distances = similarity[idx]
    
    # Sort by similarity (highest first) and get top_n
    anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
    
    # Print recommended anime names
    for i in anime_list:
        print(anime.iloc[i[0]]['Name_x'])


def main():
    df_path = r"./data/processed/data_processed.csv"
    similarity_path = r"./artifacts/similarity.pkl"
    
    anime = load_data(df_path)
    similarity = load_data(similarity_path)
    
    anime_name = input("Enter the Name of the Anime: ")
    result = recommend(anime, similarity, anime_name)

if __name__ == "__main__":
    main()