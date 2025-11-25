"""
metrics_tracker.py
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional

import pandas as pd


from app.integrations.trend_fetcher import TrendFetcher
from app.integrations.sheets_connector import append_row, read_rows
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# Config
RETRY_LIMIT = int(os.getenv("METRICS_RETRY_LIMIT", "3"))
SHEET_LOG_ENABLED = bool(os.getenv("GOOGLE_SHEET_ID"))
DEFAULT_SHEET_NAME = os.getenv("METRICS_SHEET_NAME", "daily_metrics")

# Instantiate integrations
_trends = TrendFetcher()


# -----------------------------
# Helper: safe average
# -----------------------------
def _safe_mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


# -----------------------------
# 1) Fetch and aggregate live metrics from list of post IDs 
# -----------------------------
def fetch_and_aggregate_post_metrics(post_ids: List[str]) -> Dict[str, Any]:
    """
    Aggregates post metrics.
    """
    if not post_ids:
        return {
            "total_posts": 0,
            "likes": 0,
            "replies": 0,
            "shares": 0,
            "avg_trend_score": 0.0,
            "avg_sentiment": 0.0,
            "sample_texts": []
        }

    

    likes_list = []
    replies_list = []
    shares_list = []
    trend_scores = []
    sentiment_scores = []
    sample_texts = []

    for pid in post_ids:
        try:
            text = ""  
            sample_texts.append(text)

            trend_scores.append(_trends.get_combined_trend_score(text))
            sent = analyze_sentiment(text)[0]
            sentiment_scores.append(sent.get("sentiment_score", 0.0))

            likes_list.append(0)
            replies_list.append(0)
            shares_list.append(0)

        except Exception as e:
            logger.warning(f"Failed to aggregate for post {pid}: {e}")
            continue

    return {
        "total_posts": len(post_ids),
        "likes": int(sum(likes_list)),
        "replies": int(sum(replies_list)),
        "shares": int(sum(shares_list)),
        "avg_trend_score": round(_safe_mean(trend_scores), 4),
        "avg_sentiment": round(_safe_mean(sentiment_scores), 4),
        "sample_texts": sample_texts[:5]
    }


# -----------------------------
# 2) Compute metrics from DataFrame
# -----------------------------
def compute_metrics_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "total_records": 0,
            "impressions": 0,
            "ctr": 0.0,
            "engagement_rate": 0.0,
            "conversion_rate": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 0.0,
            "avg_trend_score": 0.0,
            "avg_toxicity": 0.0,
            "dominant_emotion": "unknown"
        }

    total = len(df)

    impressions = int(df["impressions"].sum()) if "impressions" in df.columns else 0
    clicks = int(df["clicks"].sum()) if "clicks" in df.columns else 0
    likes = int(df["likes"].sum()) if "likes" in df.columns else 0
    shares = int(df["shares"].sum()) if "shares" in df.columns else 0
    comments = int(df["comments"].sum()) if "comments" in df.columns else 0
    conversions = int(df["conversions"].sum()) if "conversions" in df.columns else 0

    ctr = (clicks / impressions) if impressions > 0 else 0.0
    engagement_rate = ((likes + comments + shares) / impressions) if impressions > 0 else 0.0
    conversion_rate = (conversions / impressions) if impressions > 0 else 0.0

    pos_ratio = neg_ratio = neu_ratio = 0.0
    if "sentiment_label" in df.columns:
        labels = df["sentiment_label"].astype(str)
        pos_ratio = float((labels.str.upper() == "POSITIVE").mean())
        neg_ratio = float((labels.str.upper() == "NEGATIVE").mean())
        neu_ratio = float((labels.str.upper() == "NEUTRAL").mean())

    avg_trend = float(df["trend_score"].mean()) if "trend_score" in df.columns else 0.0
    avg_toxic = float(df["toxicity"].mean()) if "toxicity" in df.columns else 0.0

    dominant_emotion = "unknown"
    if "emotions" in df.columns:
        try:
            all_emotions = {}
            for e in df["emotions"].dropna():
                if isinstance(e, dict):
                    for k, v in e.items():
                        all_emotions[k] = all_emotions.get(k, 0) + float(v)
            if all_emotions:
                dominant_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
        except Exception:
            dominant_emotion = "unknown"

    return {
        "total_records": int(total),
        "impressions": impressions,
        "ctr": round(ctr, 4),
        "engagement_rate": round(engagement_rate, 4),
        "conversion_rate": round(conversion_rate, 4),
        "positive_ratio": round(pos_ratio, 4),
        "negative_ratio": round(neg_ratio, 4),
        "neutral_ratio": round(neu_ratio, 4),
        "avg_trend_score": round(avg_trend, 4),
        "avg_toxicity": round(avg_toxic, 4),
        "dominant_emotion": dominant_emotion
    }


# -----------------------------
# 3) Update Google Sheet
# -----------------------------
def update_google_sheet(sheet_name: str, df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    try:
        metrics = compute_metrics_from_df(df)
    except Exception as e:
        logger.error(f"Failed to compute metrics from dataframe: {e}")
        return None

    ts = pd.Timestamp.utcnow().isoformat()
    row = [
        ts,
        metrics.get("total_records", 0),
        metrics.get("impressions", 0),
        metrics.get("ctr", 0.0),
        metrics.get("engagement_rate", 0.0),
        metrics.get("conversion_rate", 0.0),
        metrics.get("positive_ratio", 0.0),
        metrics.get("negative_ratio", 0.0),
        metrics.get("neutral_ratio", 0.0),
        metrics.get("avg_trend_score", 0.0),
        metrics.get("avg_toxicity", 0.0),
        metrics.get("dominant_emotion", "unknown")
    ]

    if not SHEET_LOG_ENABLED:
        logger.info("GOOGLE_SHEET_ID not set — skipping Sheets upload.")
        return metrics

    for attempt in range(RETRY_LIMIT):
        try:
            append_row(sheet_name, row)
            logger.info(f"Metrics appended to sheet: {sheet_name}")
            return metrics
        except Exception as e:
            logger.warning(f"Append attempt {attempt+1} failed: {e}")
            time.sleep(2)

    logger.error("Failed to append metrics after retries.")
    return metrics


# -----------------------------
# 4) Public wrapper for pipeline
# -----------------------------
def push_daily_metrics(df: Optional[pd.DataFrame] = None, sheet_name: str = DEFAULT_SHEET_NAME) -> Optional[Dict[str, Any]]:
    if df is None:
        df = pd.DataFrame([{
            "impressions": 0,
            "clicks": 0,
            "likes": 0,
            "shares": 0,
            "comments": 0,
            "conversions": 0,
            "sentiment_label": "NEUTRAL",
            "trend_score": 0.0,
            "toxicity": 0.0,
            "emotions": {}
        }])

    return update_google_sheet(sheet_name, df)


# -----------------------------
# 5) CLI quick test
# -----------------------------
if __name__ == "__main__":
    data = {
        "impressions": [100, 200, 150],
        "likes": [10, 25, 5],
        "clicks": [5, 10, 4],
        "shares": [2, 3, 1],
        "comments": [1, 5, 2],
        "conversions": [1, 3, 0],
        "sentiment_label": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        "sentiment_score": [0.9, 0.2, 0.5],
        "polarity": [0.8, -0.6, 0.1],
        "toxicity": [0.01, 0.5, 0.12],
        "emotions": [
            {"joy": 0.8, "anger": 0.1},
            {"anger": 0.7, "fear": 0.1},
            {"sadness": 0.4}
        ],
        "trend_score": [55, 12, 35]
    }
    df = pd.DataFrame(data)

    print("\nComputed Metrics:")
    m = update_google_sheet("demo_metrics", df)
    print(m)

    print("\nAggregated live post metrics (fallback):")
    posts_agg = fetch_and_aggregate_post_metrics(["1234567890", "9876543210"])
    print(posts_agg)
