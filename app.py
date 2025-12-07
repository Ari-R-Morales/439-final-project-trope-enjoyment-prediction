from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import pandas as pd

# ---------- Load model, vectorizer, and trope names ----------
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

tropes_df = pd.read_csv("data/model_trope_vocabulary.csv")
trope_names = tropes_df["name"].tolist()

# ---------- Core enjoyment function (from your notebook) ----------
def pred_enjoyment(user_input_indices, book_blurb, threshold: float = 0.5):
    """
    Predict how much user enjoys book based on favorite tropes they input
    """
    # get trope probabilities from model
    X = vectorizer.transform([book_blurb])
    book_trope_probs = model.predict_proba(X)[0]

    # binary vector for user pref: 1 for liked trope, 0 for not
    user_pref = np.zeros(len(trope_names))
    user_pref[user_input_indices] = 1

    # only count tropes that are above threshold
    book_trope_present = (book_trope_probs >= threshold).astype(int)

    # overlap score
    overlap = np.dot(user_pref, book_trope_present)
    max_overlap = np.sum(user_pref)
    overlap_score = overlap / max_overlap if max_overlap > 0 else 0

    # weighted probability score
    weighted_score = (
        np.dot(user_pref, book_trope_probs) / np.sum(user_pref)
        if np.sum(user_pref) > 0
        else 0
    )

    # combined score
    combined_score = (overlap_score * 0.4) + (weighted_score * 0.6)

    # per-trope breakdown for the user's selected tropes
    trope_matches = []
    for idx in user_input_indices:
        trope_matches.append(
            {
                "trope_name": trope_names[idx],
                "probability": float(book_trope_probs[idx]),
                "present": bool(book_trope_probs[idx] >= threshold),
            }
        )

    # top tropes in the blurb overall
    top_trope_idx = np.argsort(book_trope_probs)[-5:][::-1]
    book_top_tropes = []
    for i in top_trope_idx:
        book_top_tropes.append(
            {"trope_name": trope_names[i], "probability": float(book_trope_probs[i])}
        )

    # message
    if combined_score >= 0.7:
        message = "You would LOVE this book!!"
    elif combined_score >= 0.5:
        message = "You'll probably enjoy this book!"
    elif combined_score >= 0.3:
        message = "This book could either be a hit or miss"
    else:
        message = "Maybe this book just isn't for you..."

    return {
        "overall_score": float(combined_score),
        "overlap_score": float(overlap_score),
        "weighted_score": float(weighted_score),
        "message": message,
        "user_trope_matches": trope_matches,
        "book_top_tropes": book_top_tropes,
        "total_user_tropes_found": int(overlap),
        "total_user_tropes_selected": int(max_overlap),
    }


# ---------- FastAPI setup ----------
app = FastAPI()

# Allow your frontend (Live Server / local file) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in class project this is fine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request models ----------
class EnjoymentRequest(BaseModel):
    user_trope_indices: List[int]
    book_blurb: str
    threshold: float = 0.5


# ---------- Endpoints ----------

@app.get("/tropes")
def get_tropes():
    """
    Return all tropes with their indices so the frontend can build the checklist.
    """
    return [{"index": i, "name": name} for i, name in enumerate(trope_names)]


@app.post("/predict_enjoyment")
def predict_enjoyment(req: EnjoymentRequest):
    result = pred_enjoyment(
        user_input_indices=req.user_trope_indices,
        book_blurb=req.book_blurb,
        threshold=req.threshold,
    )
    return result