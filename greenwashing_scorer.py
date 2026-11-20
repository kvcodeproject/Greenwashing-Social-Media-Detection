#!/usr/bin/env python3
"""
Greenwashing Risk Scorer for Social Media Posts
TensorFlow/Keras model + keyword ranking system.

Outputs:
- Risk level: Most Likely | Likely | Potential | Lower Risk | Least Likely
- Numerical score (0-1)
- Ranked keywords with risk category
- Supporting metrics
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Keyword dictionaries (high-risk = vague / unsubstantiated)
# ---------------------------------------------------------------------------
HIGH_RISK_KEYWORDS = {
    "eco-friendly", "ecofriendly", "eco friendly", "green", "sustainable",
    "sustainability", "natural", "all-natural", "planet-friendly", "planet friendly",
    "climate neutral", "carbon neutral", "carbon-neutral", "net zero", "net-zero",
    "climate positive", "carbon positive", "biodegradable", "compostable",
    "environmentally friendly", "environmentally-friendly", "eco", "organic",
    "clean", "pure", "kind to the planet", "earth-friendly", "earth friendly",
    "conscious", "responsible", "climate friendly", "co2 neutral", "co₂ neutral",
    "nature’s friend", "ecological", "environmentally safe", "greenwashing"  # meta
}

MEDIUM_RISK_KEYWORDS = {
    "recycled", "recyclable", "renewable", "low carbon", "low-carbon",
    "reduced emissions", "plant-based", "bio-based", "bio based",
    "offset", "offsets", "carbon offset", "carbon footprint",
    "energy efficient", "energy-efficient", "less plastic", "plastic free",
    "plastic-free", "zero waste", "zero-waste"
}

LOW_RISK_POSITIVE = {  # evidence / specificity signals (reduce risk)
    "certified", "certification", "fsc", "gots", "usda organic", "b corp",
    "b-corp", "energy star", "iso 14001", "sbti", "ghg protocol",
    "scope 1", "scope 2", "scope 3", "third-party", "third party",
    "verified", "independent verification", "life cycle assessment", "lca",
    "percent", "%", "reduction of", "reduced by", "since 20", "by 2030",
    "by 2025", "tco2e", "kg co2"
}

HEDGING = {"aims to", "strives to", "plans to", "committed to", "working towards",
           "hoping to", "intends to", "goal of"}

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(text: str) -> dict:
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = max(len(words), 1)

    high = [k for k in HIGH_RISK_KEYWORDS if k in text_lower]
    medium = [k for k in MEDIUM_RISK_KEYWORDS if k in text_lower]
    positive = [k for k in LOW_RISK_POSITIVE if k in text_lower]
    hedge = [h for h in HEDGING if h in text_lower]

    # Numbers / percentages
    has_percent = bool(re.search(r'\d+\s*%|\d+\s*percent', text_lower))
    has_number = bool(re.search(r'\d+', text))
    has_year = bool(re.search(r'20\d{2}', text))

    return {
        "high_risk_count": len(high),
        "medium_risk_count": len(medium),
        "positive_evidence_count": len(positive),
        "hedging_count": len(hedge),
        "high_risk_density": len(high) / total_words,
        "has_percent": int(has_percent),
        "has_specific_number": int(has_number),
        "has_year_or_timeline": int(has_year),
        "text_length": len(text),
        "high_risk_terms": high,
        "medium_risk_terms": medium,
        "positive_terms": positive,
        "hedging_terms": hedge,
    }

def features_to_vector(feat: dict) -> np.ndarray:
    return np.array([
        feat["high_risk_count"],
        feat["medium_risk_count"],
        feat["positive_evidence_count"],
        feat["hedging_count"],
        feat["high_risk_density"] * 100,
        feat["has_percent"],
        feat["has_specific_number"],
        feat["has_year_or_timeline"],
        min(feat["text_length"] / 100, 10),
    ], dtype=np.float32)

# ---------------------------------------------------------------------------
# Synthetic training data (illustrative)
# ---------------------------------------------------------------------------
def generate_synthetic_data(n=800):
    high_examples = [
        "Our products are 100% eco-friendly and sustainable! #GreenLiving",
        "Proud to be a carbon neutral company. Together we can save the planet 🌍",
        "Natural, pure, and kind to the planet. Shop our green collection today!",
        "We're committed to a net-zero future. Eco-friendly fashion for everyone.",
        "Biodegradable packaging + sustainable materials = better planet.",
        "Join us in our climate positive journey. #SustainableLiving",
        "All-natural ingredients that are environmentally friendly.",
        "Green energy, green products, green everything. We're eco!",
    ]
    medium_examples = [
        "We've reduced plastic in our packaging by using more recycled materials.",
        "Switching to renewable energy in our factories. Low-carbon production.",
        "Plant-based materials in our new line. Working towards zero waste.",
        "Our carbon footprint is smaller this year thanks to offsets.",
    ]
    low_examples = [
        "Reduced Scope 1 & 2 emissions by 18% since 2022 (GHG Protocol, third-party verified).",
        "Packaging is 85% post-consumer recycled cardboard, FSC certified.",
        "GOTS certified organic cotton. Water use reduced 71% vs conventional.",
        "100% of operational electricity from wind & solar (Guarantee of Origin 2025).",
        "SBTi-validated target: 35% absolute reduction by 2030. Interim 2025 milestone met.",
    ]
    neutral = [
        "New product launch next week! Check our website for details.",
        "Thanks for the amazing feedback on our latest campaign.",
        "Team outing this Friday. Looking forward to it!",
        "Customer service hours updated. DM us for support.",
    ]

    posts, labels = [], []
    for _ in range(n // 4):
        posts.append(np.random.choice(high_examples) + " " + np.random.choice(["", "#Eco", "Buy now!"]))
        labels.append(0.85 + np.random.uniform(-0.1, 0.1))  # high risk
    for _ in range(n // 4):
        posts.append(np.random.choice(medium_examples))
        labels.append(0.45 + np.random.uniform(-0.1, 0.1))
    for _ in range(n // 4):
        posts.append(np.random.choice(low_examples))
        labels.append(0.15 + np.random.uniform(-0.05, 0.1))
    for _ in range(n // 4):
        posts.append(np.random.choice(neutral))
        labels.append(0.05 + np.random.uniform(0, 0.05))

    return posts, np.clip(labels, 0, 1)

# ---------------------------------------------------------------------------
# Build & train model
# ---------------------------------------------------------------------------
def build_model(input_dim=9):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")  # risk score 0-1
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_model():
    posts, labels = generate_synthetic_data(1000)
    X = np.array([features_to_vector(extract_features(p)) for p in posts])
    y = np.array(labels, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model()
    model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.15, verbose=0)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MAE: {mae:.4f}")

    return model, scaler

# ---------------------------------------------------------------------------
# Scoring & ranking
# ---------------------------------------------------------------------------
def risk_level(score: float) -> str:
    if score >= 0.70:
        return "Most Likely (High Risk)"
    elif score >= 0.50:
        return "Likely"
    elif score >= 0.30:
        return "Potential"
    elif score >= 0.15:
        return "Lower Risk"
    else:
        return "Least Likely"

def rank_keywords(feat: dict) -> list:
    ranked = []
    for term in feat["high_risk_terms"]:
        ranked.append({"keyword": term, "risk": "High Risk", "weight": 1.0})
    for term in feat["medium_risk_terms"]:
        ranked.append({"keyword": term, "risk": "Medium / Potential", "weight": 0.6})
    for term in feat["positive_terms"]:
        ranked.append({"keyword": term, "risk": "Evidence (Lowers Risk)", "weight": -0.5})
    for term in feat["hedging_terms"]:
        ranked.append({"keyword": term, "risk": "Hedging (Increases Uncertainty)", "weight": 0.4})
    # sort by absolute weight descending
    ranked.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return ranked

def score_post(text: str, model, scaler) -> dict:
    feat = extract_features(text)
    vec = features_to_vector(feat).reshape(1, -1)
    vec_scaled = scaler.transform(vec)
    score = float(model.predict(vec_scaled, verbose=0)[0][0])

    # Adjust score slightly with rule-based boost for pure high-risk
    if feat["high_risk_count"] >= 2 and feat["positive_evidence_count"] == 0:
        score = min(1.0, score + 0.1)

    return {
        "post": text,
        "greenwashing_score": round(score, 3),
        "risk_level": risk_level(score),
        "metrics": {
            "high_risk_keyword_count": feat["high_risk_count"],
            "medium_risk_keyword_count": feat["medium_risk_count"],
            "evidence_signal_count": feat["positive_evidence_count"],
            "hedging_count": feat["hedging_count"],
            "high_risk_density_per_100_words": round(feat["high_risk_density"] * 100, 2),
            "has_quantitative_claim": bool(feat["has_percent"] or feat["has_specific_number"]),
            "has_timeline": bool(feat["has_year_or_timeline"]),
        },
        "ranked_keywords": rank_keywords(feat),
    }

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Training greenwashing risk model...")
    model, scaler = train_model()

    # Save
    out_dir = Path("/workspace/artifacts/greenwashing_model")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    # Save scaler parameters (simple)
    np.save(out_dir / "scaler_mean.npy", scaler.mean_)
    np.save(out_dir / "scaler_scale.npy", scaler.scale_)

    print(f"Model saved to {out_dir}")

    # Demo posts
    demo_posts = [
        "Our brand is 100% eco-friendly and sustainable! Carbon neutral fashion for a better planet. #Green",
        "Reduced Scope 1 and 2 emissions by 22% since 2021. Verified by independent third-party audit (GHG Protocol).",
        "New collection is natural, pure, and kind to the earth. Shop sustainable today!",
        "Excited to announce our partnership with local artists. Check out the new designs!",
        "We're working towards net zero by 2050 and using more recycled materials in packaging.",
    ]

    results = []
    for p in demo_posts:
        res = score_post(p, model, scaler)
        results.append(res)
        print("\n" + "="*60)
        print(f"POST: {p[:80]}...")
        print(f"SCORE: {res['greenwashing_score']}  →  {res['risk_level']}")
        print("KEYWORDS:")
        for kw in res["ranked_keywords"][:5]:
            print(f"  - {kw['keyword']} ({kw['risk']})")
        print("METRICS:", json.dumps(res["metrics"], indent=2))

    # Save demo results
    with open(out_dir / "demo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Done. Model + demo results saved.")
