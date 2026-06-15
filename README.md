# 🎵 Music Genre Classifier

A machine learning web app that predicts the **genre** and **emotion** of a song — either by uploading an audio file or by entering Spotify-style audio features manually.

---

## 📁 Project Structure

```
C:\Music_Classifier\
├── app.py                  ← FastAPI backend (genre + emotion prediction)
├── validmusic.py           ← Pydantic input validation
├── musicclassifier.pkl     ← Trained Random Forest model
├── scaler.pkl              ← StandardScaler (fitted on training data)
├── genre_map.pkl           ← Class number → genre name mapping
├── index.html              ← Frontend (dashboard + upload + manual form)
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## 🚀 Quick Start (Local)

**Step 1 — Install dependencies**
```powershell
cd "C:\Music_Classifier"
python -m pip install -r requirements.txt
```

**Step 2 — Start the API**
```powershell
python app.py
```
You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:5001
```

**Step 3 — Open the app**

Double-click `index.html` in File Explorer to open it in your browser.

---

## 🤖 Model Details

| Item | Value |
|------|-------|
| Algorithm | Random Forest Classifier |
| Training songs | 17,996 |
| Features | 14 Spotify audio features |
| Classes | 11 genres |
| Test accuracy | ~50.7% |
| Class balancing | `class_weight='balanced'` |

### Genre mapping (Class → Genre)

| Class | Genre |
|-------|-------|
| 0 | Acoustic/Folk |
| 1 | Alt_Music |
| 2 | Blues |
| 3 | Bollywood |
| 4 | Country |
| 5 | HipHop |
| 6 | Indie Alt |
| 7 | Instrumental |
| 8 | Metal |
| 9 | Pop |
| 10 | Rock |

---

## 🎭 Emotion Classifier (Rule-Based)

Emotion is detected from audio features using Russell's Circumplex Model:

| Emotion | Condition |
|---------|-----------|
| 🔥 Euphoric | High valence + High energy + High speechiness |
| 🎉 Excited | High valence + High energy + Fast tempo |
| 😊 Happy | High valence + High energy |
| 😌 Peaceful | High valence + Low energy + Acoustic |
| 🌙 Dreamy | High valence + Low energy + Instrumental |
| 😎 Relaxed | High valence + Low energy |
| 😤 Angry | Low valence + Very high energy + Fast tempo |
| 😰 Tense | Low valence + High energy |
| 😢 Sad | Low valence + Low energy + Very low valence |
| 🥺 Melancholic | Low valence + Low energy |

---

## 🔌 API Endpoints

### `GET /`
Health check — returns API status and list of genres.

```json
{
  "message": "Music Genre Classifier API is running!",
  "genres": ["Acoustic/Folk", "Alt_Music", "Blues", ...]
}
```

---

### `POST /predict`
Predict genre + emotion from 14 manual audio features.

**Request body:**
```json
{
  "Popularity": 65,
  "danceability": 0.78,
  "energy": 0.72,
  "key": 5,
  "loudness": -5.5,
  "mode": 1,
  "speechiness": 0.05,
  "acousticness": 0.15,
  "instrumentalness": 0.001,
  "liveness": 0.12,
  "valence": 0.65,
  "tempo": 120,
  "duration_in min/ms": 210000,
  "time_signature": 4
}
```

**Response:**
```json
{
  "predicted_class": 10,
  "predicted_genre": "Rock",
  "confidence": "62.5%",
  "top3": [
    {"genre": "Rock",     "confidence": "62.5%"},
    {"genre": "Metal",    "confidence": "18.3%"},
    {"genre": "Indie Alt","confidence": "9.1%"}
  ],
  "emotion": {
    "emotion": "Excited",
    "emoji": "🎉",
    "color": "#eab308",
    "description": "High tempo, uplifting and celebratory"
  }
}
```

---

### `POST /predict-file`
Predict genre + emotion by uploading a song file (`.mp3`, `.wav`, `.flac`, `.ogg`).

**Form data:** `file` — the audio file

**Response:** Same format as `/predict`, plus `"filename"` field.

> Audio features are extracted using `scipy` + `miniaudio` (no librosa/numba needed — works on Python 3.14).

---

## 📦 Dependencies

Install all at once:
```powershell
python -m pip install -r requirements.txt
```

Or manually:
```powershell
python -m pip install fastapi uvicorn scikit-learn numpy pandas scipy joblib miniaudio soundfile python-multipart
```

> **Note:** This project intentionally avoids `librosa`, `numba`, and `llvmlite` because they are not compatible with Python 3.14. Audio feature extraction is handled by `scipy` + `miniaudio` instead.

---

## 🌐 Deployment

### Backend → Render.com

1. Create `Procfile` (no extension):
   ```
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

2. Push to GitHub:
   ```powershell
   git init
   git add .
   git commit -m "music classifier"
   git branch -M main
   git remote add origin https://github.com/yourname/music-classifier.git
   git push -u origin main
   ```

3. Go to [render.com](https://render.com) → New Web Service → Connect repo → Deploy.

4. You get a live URL like: `https://music-classifier.onrender.com`

### Frontend → Netlify

1. Go to [netlify.com](https://netlify.com)
2. Drag and drop `index.html`
3. Update the API URL in index.html from `http://127.0.0.1:5001` to your Render URL

---

## ⚠️ Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `pip not recognized` | Use `python -m pip install ...` instead |
| `Port 5001 already in use` | Run `taskkill /IM python.exe /F` then restart |
| `Cannot reach API` | Make sure `python app.py` is running first |
| `CORS error in browser` | Confirm `CORSMiddleware` is in `app.py` |
| `numba` / `llvmlite` errors | Uninstall both: `python -m pip uninstall numba llvmlite -y` |
| `No module named pkg_resources` | Run `python -m pip install setuptools` |
| `musicclassifier.pkl not found` | Run your training notebook first to generate `.pkl` files |
| `.pkl` too large for GitHub | Retrain with `n_estimators=50` to reduce file size |

---

## 🗺️ Architecture

```
User browser
    ↓ opens
index.html (Netlify / local file)
    ↓ POST /predict or /predict-file
app.py — FastAPI (http://127.0.0.1:5001 or Render)
    ↓ loads
musicclassifier.pkl + scaler.pkl + genre_map.pkl
    ↓ returns
{ genre, confidence, top3, emotion }
    ↓
index.html shows result + emotion card
```

---

## 📊 Training Data Distribution

| Genre | Samples |
|-------|---------|
| Rock | 4,949 |
| Indie Alt | 2,587 |
| Pop | 2,524 |
| Metal | 1,854 |
| HipHop | 1,447 |
| Alt_Music | 1,373 |
| Blues | 1,272 |
| Acoustic/Folk | 625 |
| Instrumental | 576 |
| Bollywood | 402 |
| Country | 387 |

---

*Built with FastAPI · scikit-learn · scipy · miniaudio*
