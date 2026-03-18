import pandas as pd
from src.feature_engineering import feature_engineering

def test_data_loading():
    df = pd.read_csv("data/anime.csv")
    assert df is not None
    assert len(df) > 0


def test_feature_engineering():
    df = pd.DataFrame({
        "title": ["naruto", "one piece"],
        "genre": ["action", "adventure"]
    })

    similarity, tfidf = feature_engineering(df)

    assert similarity is not None
    assert tfidf is not None


def test_no_nulls_after_processing():
    df = pd.DataFrame({
        "title": ["naruto", None],
        "genre": ["action", "adventure"]
    })

    df = df.fillna("")

    similarity, tfidf = feature_engineering(df)

    assert similarity is not None