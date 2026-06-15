import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Only retrain if pkl files are missing
if os.path.exists("musicclassifier.pkl"):
    print("Model already exists — skipping training")
else:
    print("Training model...")
    df = pd.read_csv("train_cleaned.csv")

    FEATURE_COLS = [
        'Popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'duration_in min/ms', 'time_signature'
    ]

    X = df[FEATURE_COLS].values
    y = df['Class'].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=50,          # ← keep small so pkl stays under 100MB
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_scaled, y)

    CLASS_TO_GENRE = {
        0:"Acoustic/Folk", 1:"Alt_Music", 2:"Blues", 3:"Bollywood",
        4:"Country", 5:"HipHop", 6:"Indie Alt", 7:"Instrumental",
        8:"Metal", 9:"Pop", 10:"Rock"
    }

    joblib.dump(model,          "musicclassifier.pkl")
    joblib.dump(scaler,         "scaler.pkl")
    joblib.dump(CLASS_TO_GENRE, "genre_map.pkl")
    print("Training done!")