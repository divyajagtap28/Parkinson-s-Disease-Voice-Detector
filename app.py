import io
import os
import pickle
import tempfile
from pathlib import Path

import librosa
import numpy as np
import scipy.stats
import streamlit as st
from pydub import AudioSegment

st.set_page_config(page_title="Parkinson's Voice Detector", page_icon="🎤", layout="centered")

MODEL_PATH = Path(__file__).parent / "ml" / "best_pd_model.pkl"
THRESHOLD = 0.7
MIN_SECONDS = 3


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["scaler"], data["selected_features"]


def to_wav_bytes(raw: bytes, ext: str) -> bytes:
    if ext == "wav":
        return raw
    audio = AudioSegment.from_file(io.BytesIO(raw), format=ext)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def extract_features(y: np.ndarray, sr: int) -> dict:
    f = {}
    pitches = librosa.yin(y, fmin=75, fmax=600)
    pitches = pitches[~np.isnan(pitches)]
    f["Fo"] = float(np.mean(pitches)) if len(pitches) > 0 else 0.0
    if len(pitches) > 1:
        f["Jitter"] = float(np.mean(np.abs(np.diff(pitches))) / (f["Fo"] + 1e-6))
    else:
        f["Jitter"] = 0.0

    rms = librosa.feature.rms(y=y)[0]
    if len(rms) > 1:
        shimmer = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-6))
        f["Shimmer"] = shimmer
        f["Shimmer(dB)"] = 20 * np.log10(shimmer + 1e-6)
        f["Shimmer:APQ5"] = shimmer * 0.8
        f["Shimmer:APQ11"] = shimmer * 0.6
    else:
        f["Shimmer"] = f["Shimmer(dB)"] = f["Shimmer:APQ5"] = f["Shimmer:APQ11"] = 0.0

    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    f["HNR"] = float(np.mean(harmonic) / (np.mean(percussive) + 1e-6))
    f["NHR"] = 1.0 / (f["HNR"] + 1e-6)
    f["RPDE"] = float(scipy.stats.entropy(np.abs(librosa.stft(y).mean(axis=0)) + 1e-6))
    f["DFA"] = float(np.mean(np.abs(np.diff(y))))
    f["PPE"] = float(scipy.stats.entropy(np.abs(y) + 1e-6))
    f["spread1"] = float(np.std(y))
    f["spread2"] = float(scipy.stats.kurtosis(y))
    return f


def predict(raw: bytes, ext: str):
    model, scaler, selected = load_model()
    wav = to_wav_bytes(raw, ext)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav)
        path = tmp.name
    try:
        y, sr = librosa.load(path, sr=22050)
    finally:
        os.remove(path)
    if len(y) < sr * MIN_SECONDS:
        raise ValueError(f"Recording too short (minimum {MIN_SECONDS} seconds).")
    feats = extract_features(y, sr)
    X = np.array([feats[k] for k in selected]).reshape(1, -1)
    proba = model.predict_proba(scaler.transform(X))[0]
    return float(proba[0]), float(proba[1]), {k: feats[k] for k in selected}


st.title("🎤 Parkinson's Disease Voice Detector")
st.caption("Record or upload a sustained vowel sound (e.g. 'aaah') for at least 3 seconds.")
st.warning("Educational demo only. This is NOT a medical diagnosis. Consult a doctor for health concerns.")

tab_rec, tab_up = st.tabs(["🎙️ Record", "📁 Upload"])
audio_bytes, ext = None, "wav"

with tab_rec:
    rec = st.audio_input("Record your voice")
    if rec is not None:
        audio_bytes, ext = rec.getvalue(), "wav"

with tab_up:
    up = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg", "flac", "m4a", "aac", "webm"])
    if up is not None:
        audio_bytes, ext = up.getvalue(), up.name.rsplit(".", 1)[-1].lower()
        st.audio(audio_bytes)

if st.button("Analyze", type="primary", disabled=audio_bytes is None):
    with st.spinner("Extracting features and running SVM model..."):
        try:
            healthy_p, pd_p, used = predict(audio_bytes, ext)
        except Exception as e:
            st.error(f"Could not analyze audio: {e}")
        else:
            is_pd = pd_p >= THRESHOLD
            conf = pd_p if is_pd else healthy_p
            if is_pd:
                st.error(f"Potential Parkinson's detected (confidence {conf:.1%})")
            else:
                st.success(f"No signs of Parkinson's detected (confidence {conf:.1%})")
            if conf < THRESHOLD:
                st.info("Low confidence result. Try a longer, clearer recording.")
            c1, c2 = st.columns(2)
            c1.metric("Healthy probability", f"{healthy_p:.1%}")
            c2.metric("Parkinson probability", f"{pd_p:.1%}")
            with st.expander("Features used by the model"):
                st.json({k: round(v, 5) for k, v in used.items()})
