import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from validation_music import MusicFeatures
import joblib
import numpy as np
import tempfile
from scipy.io import wavfile
from scipy import signal

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Music Genre Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts ───────────────────────────────────────────────────────
model          = joblib.load("musicclassifier.pkl")
scaler         = joblib.load("scaler.pkl")
CLASS_TO_GENRE = joblib.load("genre_map.pkl")

FEATURE_COLS = [
    'Popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_in min/ms', 'time_signature'
]


def predict_emotion(valence, energy, tempo,
                    acousticness, speechiness, instrumentalness):
    if valence >= 0.6 and energy >= 0.6:
        if tempo > 130:
            return {"emotion":"Excited",     "emoji":"🎉", "description":"High energy, fast tempo, very positive feel", "color":"#f59e0b"}
        elif speechiness > 0.15:
            return {"emotion":"Euphoric",    "emoji":"🔥", "description":"Energetic with strong vocal presence",        "color":"#ef4444"}
        else:
            return {"emotion":"Happy",       "emoji":"😊", "description":"Positive, upbeat and feel-good",              "color":"#eab308"}
    elif valence >= 0.6 and energy < 0.6:
        if acousticness > 0.6:
            return {"emotion":"Peaceful",    "emoji":"😌", "description":"Soft, acoustic and serene",                   "color":"#10b981"}
        elif instrumentalness > 0.5:
            return {"emotion":"Dreamy",      "emoji":"🌙", "description":"Instrumental, floating and atmospheric",      "color":"#8b5cf6"}
        else:
            return {"emotion":"Relaxed",     "emoji":"😎", "description":"Easygoing, pleasant and chill",              "color":"#06b6d4"}
    elif valence < 0.6 and energy >= 0.6:
        if energy > 0.85 and tempo > 140:
            return {"emotion":"Angry",       "emoji":"😤", "description":"Intense, aggressive and powerful",            "color":"#dc2626"}
        elif speechiness > 0.20:
            return {"emotion":"Defiant",     "emoji":"✊", "description":"Strong vocal delivery with attitude",         "color":"#ea580c"}
        else:
            return {"emotion":"Tense",       "emoji":"😰", "description":"Driving energy with dark undertones",        "color":"#7c3aed"}
    else:
        if valence < 0.3:
            return {"emotion":"Sad",         "emoji":"😢", "description":"Low energy, melancholic and emotional",      "color":"#3b82f6"}
        elif acousticness > 0.6:
            return {"emotion":"Melancholic", "emoji":"🥺", "description":"Reflective, acoustic and bittersweet",       "color":"#6366f1"}
        else:
            return {"emotion":"Gloomy",      "emoji":"🌧️","description":"Dark mood with subdued energy",              "color":"#64748b"}


def librosa_to_spotify_features(file_path):

    if file_path.lower().endswith('.mp3'):
        try:
            import miniaudio
            import struct
            decoded   = miniaudio.decode_file(file_path,
                            output_format=miniaudio.SampleFormat.SIGNED16,
                            nchannels=1, sample_rate=22050)
            wav_path  = file_path.replace('.mp3', '.wav')
            # Write as wav manually
            num_frames   = len(decoded.samples)
            num_channels = 1
            sampwidth    = 2   # 16-bit
            framerate    = 22050
            with open(wav_path, 'wb') as wf:
                # WAV header
                data_size = num_frames * sampwidth
                wf.write(b'RIFF')
                wf.write(struct.pack('<I', 36 + data_size))
                wf.write(b'WAVE')
                wf.write(b'fmt ')
                wf.write(struct.pack('<IHHIIHH', 16, 1, num_channels,
                                     framerate, framerate * sampwidth,
                                     sampwidth, 16))
                wf.write(b'data')
                wf.write(struct.pack('<I', data_size))
                wf.write(bytes(decoded.samples))
            file_path = wav_path
        except ImportError:
            raise ValueError("Run: pip install miniaudio")
        except Exception as e:
            raise ValueError(f"Could not decode mp3: {e}")

    # ── Load wav ───────────────────────────────────────────────────────────────
    sr, y = wavfile.read(file_path)
    if y.ndim > 1:
        y = y.mean(axis=1)                          # stereo → mono
    y = y.astype(np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9)             # normalize -1 to 1

    # Trim to 30 seconds
    if len(y) > sr * 30:
        y = y[:sr * 30]

    # ── Features ───────────────────────────────────────────────────────────────
    rms         = np.sqrt(np.mean(y**2))
    energy      = float(np.clip(rms * 10, 0, 1))
    loudness    = float(np.clip(20 * np.log10(rms + 1e-9), -60, 0))

    zcr         = np.sum(np.diff(np.sign(y)) != 0) / len(y)
    speechiness = float(np.clip(zcr * 10, 0, 1))

    freqs       = np.fft.rfftfreq(len(y), d=1/sr)
    fft         = np.abs(np.fft.rfft(y))
    fft_sum     = fft.sum() + 1e-9

    centroid     = float(np.sum(freqs * fft) / fft_sum)
    acousticness = float(np.clip(1 - (centroid / 8000), 0, 1))

    low_energy  = fft[(freqs >= 20)  & (freqs < 250)].sum()
    mid_energy  = fft[(freqs >= 250) & (freqs < 4000)].sum()
    valence     = float(np.clip(mid_energy / (low_energy + mid_energy + 1e-9), 0, 1))

    flatness         = float(np.exp(np.mean(np.log(fft + 1e-9))) / (np.mean(fft) + 1e-9))
    instrumentalness = float(np.clip(flatness * 2, 0, 1))

    frame_size = int(sr * 0.023)
    hop        = int(sr * 0.010)
    n_frames   = (len(y) - frame_size) // hop
    envelope   = np.array([np.sqrt(np.mean(y[i*hop:i*hop+frame_size]**2))
                            for i in range(n_frames)])
    corr       = np.correlate(envelope, envelope, mode='full')
    corr       = corr[len(corr)//2:]
    min_lag    = max(1, int(60 / 200 * (sr / hop)))
    max_lag    = min(int(60 / 60 * (sr / hop)), len(corr)-1)
    peak_lag   = np.argmax(corr[min_lag:max_lag]) + min_lag
    tempo      = float(np.clip(60 / (peak_lag * hop / sr), 40, 220)) if peak_lag > 0 else 120.0

    beat_var     = float(np.std(envelope))
    danceability = float(np.clip(1 - beat_var * 5, 0, 1))

    high_energy = fft[freqs > 4000].sum()
    liveness    = float(np.clip(high_energy / fft_sum, 0, 1))

    dominant_freq = freqs[np.argmax(fft)]
    key  = int((12 * np.log2(dominant_freq / 27.5 + 1e-9)) % 12) if dominant_freq > 0 else 0
    key  = int(np.clip(key, 0, 11))
    mode = 1 if mid_energy > low_energy else 0

    duration_ms = int(len(y) / sr * 1000)

    features = [
        50, danceability, energy, key, loudness, mode,
        speechiness, acousticness, instrumentalness, liveness,
        valence, tempo, duration_ms, 4
    ]

    emotion_inputs = dict(
        valence=valence, energy=energy, tempo=tempo,
        acousticness=acousticness, speechiness=speechiness,
        instrumentalness=instrumentalness
    )

    return features, emotion_inputs


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/")
def home():
    return {
        "message": "Music Genre Classifier API is running!",
        "genres":  list(CLASS_TO_GENRE.values()),
        "docs":    "/docs"
    }


@app.post("/predict")
def predict_genre(data: MusicFeatures):
    d               = data.dict(by_alias=True)
    features        = np.array([[d[col] for col in FEATURE_COLS]])
    features_scaled = scaler.transform(features)
    pred_class      = int(model.predict(features_scaled)[0])
    genre_name      = CLASS_TO_GENRE[pred_class]
    proba           = model.predict_proba(features_scaled)[0]
    top3_idx        = np.argsort(proba)[::-1][:3]
    top3 = [
        {"genre": CLASS_TO_GENRE[i], "confidence": f"{proba[i]*100:.1f}%"}
        for i in top3_idx
    ]
    emotion = predict_emotion(
        valence=d['valence'], energy=d['energy'], tempo=d['tempo'],
        acousticness=d['acousticness'], speechiness=d['speechiness'],
        instrumentalness=d['instrumentalness']
    )
    return {
        "predicted_class": pred_class,
        "predicted_genre": genre_name,
        "confidence":      f"{proba[pred_class]*100:.1f}%",
        "top3":            top3,
        "emotion":         emotion
    }


@app.post("/predict-file")
async def predict_from_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        features, emotion_inputs = librosa_to_spotify_features(tmp_path)
        features_scaled = scaler.transform([features])
        pred_class      = int(model.predict(features_scaled)[0])
        genre_name      = CLASS_TO_GENRE[pred_class]
        proba           = model.predict_proba(features_scaled)[0]
        top3_idx        = np.argsort(proba)[::-1][:3]
        top3 = [
            {"genre": CLASS_TO_GENRE[i], "confidence": f"{proba[i]*100:.1f}%"}
            for i in top3_idx
        ]
        emotion = predict_emotion(**emotion_inputs)
        return {
            "filename":        file.filename,
            "predicted_class": pred_class,
            "predicted_genre": genre_name,
            "confidence":      f"{proba[pred_class]*100:.1f}%",
            "top3":            top3,
            "emotion":         emotion
        }
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5001)