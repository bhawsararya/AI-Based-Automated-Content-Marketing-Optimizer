# sentiment_analyzer.py

import logging
from typing import List, Union, Dict
from textblob import TextBlob

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

try:
    from langdetect import detect
    LANG_AVAILABLE = True
except:
    LANG_AVAILABLE = False

from app.integrations.trend_fetcher import TrendFetcher
from app.integrations.sheets_connector import append_row

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_senti_model = None
_emotion_model = None


def _init_sentiment_model():
    return pipeline("sentiment-analysis")


def _init_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )


def detect_language(text: str) -> str:
    if not LANG_AVAILABLE:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def fallback_sentiment(text: str) -> Dict:
    polarity = TextBlob(text).sentiment.polarity

    if polarity >= 0.05:
        label = "POSITIVE"
    elif polarity <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "label": label,
        "score": abs(polarity),
        "polarity": polarity,
    }


def simplify_emotion_output(raw_output: List[Dict]) -> Dict:
    return {x["label"]: float(x["score"]) for x in raw_output}


def analyze_sentiment(texts: Union[str, List[str]]) -> List[Dict]:
    if isinstance(texts, str):
        texts = [texts]

    global _senti_model, _emotion_model

    if HF_AVAILABLE:
        if _senti_model is None:
            _senti_model = _init_sentiment_model()
        if _emotion_model is None:
            _emotion_model = _init_emotion_model()

    trend_engine = TrendFetcher()
    results = []

    for text in texts:
        lang = detect_language(text)

        try:
            if HF_AVAILABLE:
                pred = _senti_model(text)[0]
                label = pred["label"].upper()
                score = float(pred["score"])
                polarity = TextBlob(text).sentiment.polarity
            else:
                raise Exception("HF not available")
        except:
            fallback = fallback_sentiment(text)
            label = fallback["label"]
            score = fallback["score"]
            polarity = fallback["polarity"]

        norm_score = 1 - score if label.startswith("NEG") else score

        emotions = {}
        if HF_AVAILABLE:
            try:
                raw = _emotion_model(text)[0]
                emotions = simplify_emotion_output(raw)
            except:
                emotions = {}

        trend_score = trend_engine.get_combined_trend_score(text)

        entry = {
            "text": text,
            "sentiment_label": label,
            "sentiment_score": round(norm_score, 4),
            "polarity": polarity,
            "emotions": emotions,
            "language": lang,
            "trend_score": trend_score,
        }

        try:
            append_row("sentiment_results", [
                text[:80] + "...",
                label,
                norm_score,
                polarity,
                trend_score,
            ])
        except:
            pass

        results.append(entry)

    return results


def analyze_from_dataframe(df, text_column: str):
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    out = analyze_sentiment(df[text_column].tolist())

    df["sentiment_label"] = [r["sentiment_label"] for r in out]
    df["sentiment_score"] = [r["sentiment_score"] for r in out]
    df["polarity"] = [r["polarity"] for r in out]
    df["emotions"] = [r["emotions"] for r in out]
    df["language"] = [r["language"] for r in out]
    df["trend_score"] = [r["trend_score"] for r in out]

    return df
