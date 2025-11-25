# metrics_hub.py 

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd

# Local CSV paths
DATA_DIR = "data"
CAMPAIGNS_CSV = os.path.join(DATA_DIR, "campaigns1.csv")
HISTORICAL_CSV = os.path.join(DATA_DIR, "historical_metrics.csv")

os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

try:
    from app.integrations.sheets_connector import append_row
    SHEETS_AVAILABLE = True
except Exception:
    append_row = None
    SHEETS_AVAILABLE = False
    logger.info("integrations.sheets_connector not available — Sheets disabled.")

try:
    from app.integrations.trend_fetcher import TrendFetcher
    _TREND_AVAILABLE = True
    _TRENDER = TrendFetcher()
except Exception as e:
    _TREND_AVAILABLE = False
    _TRENDER = None
    logger.info(f"TrendFetcher not available: {e}")


from app.sentiment_engine.sentiment_analyzer import analyze_sentiment


GOOGLE_SHEET_ENABLED = bool(os.getenv("GOOGLE_SHEET_ID")) and SHEETS_AVAILABLE


def _init_file(path: str, columns: List[str]):
    if not os.path.exists(path):
        logger.info(f"Creating CSV: {path}")
        pd.DataFrame(columns=columns).to_csv(path, index=False)


_init_file(CAMPAIGNS_CSV, [
    "timestamp", "campaign_id", "variant", "post_id", "platform",
    "impressions", "clicks", "conversions", "ctr", "conv_rate",
    "sentiment", "trend_score", "polarity"
])

_init_file(HISTORICAL_CSV, [
    "timestamp", "campaign_id", "variant", "ctr", "sentiment",
    "polarity", "conversions", "trend_score"
])


def record_campaign_metrics(
    campaign_id: str,
    variant: str,
    impressions: int,
    clicks: int,
    conversions: int,
    sentiment_score: float,
    trend_score: float = 0.0,
    platform: str = "unknown",
    post_id: Optional[str] = None
) -> None:

    timestamp = datetime.utcnow().isoformat()
    ctr = (clicks / impressions) if impressions > 0 else 0.0
    conv_rate = (conversions / clicks) if clicks > 0 else 0.0

    new_row = {
        "timestamp": timestamp,
        "campaign_id": campaign_id,
        "variant": variant,
        "post_id": post_id or "",
        "platform": platform,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "ctr": ctr,
        "conv_rate": conv_rate,
        "sentiment": float(sentiment_score),
        "trend_score": float(trend_score),
        "polarity": float(sentiment_score)
    }

    try:
        df = pd.read_csv(CAMPAIGNS_CSV)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(CAMPAIGNS_CSV, index=False)
        logger.info(f"Recorded campaign metrics: {campaign_id} / {variant}")
    except Exception as e:
        logger.error(f"Failed to write to {CAMPAIGNS_CSV}: {e}")

    try:
        hist_row = {
            "timestamp": timestamp,
            "campaign_id": campaign_id,
            "variant": variant,
            "ctr": ctr,
            "sentiment": float(sentiment_score),
            "polarity": float(sentiment_score),
            "conversions": int(conversions),
            "trend_score": float(trend_score)
        }
        hist = pd.read_csv(HISTORICAL_CSV)
        hist = pd.concat([hist, pd.DataFrame([hist_row])], ignore_index=True)
        hist.to_csv(HISTORICAL_CSV, index=False)
    except Exception as e:
        logger.error(f"Failed to write to {HISTORICAL_CSV}: {e}")

    if GOOGLE_SHEET_ENABLED:
        try:
            append_row("campaigns", [
                timestamp,
                campaign_id,
                variant,
                post_id or "",
                platform,
                impressions,
                clicks,
                conversions,
                round(ctr, 6),
                round(conv_rate, 6),
                round(float(sentiment_score), 4),
                round(float(trend_score), 4)
            ])
        except Exception as e:
            logger.warning(f"Failed to append campaign to Sheets: {e}")


def fetch_recent_metrics(limit: int = 50) -> pd.DataFrame:
    try:
        df = pd.read_csv(CAMPAIGNS_CSV)
        return df.tail(limit)
    except Exception as e:
        logger.error(f"Failed to read {CAMPAIGNS_CSV}: {e}")
        return pd.DataFrame()


def fetch_campaign_history(campaign_id: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(CAMPAIGNS_CSV)
        return df[df["campaign_id"] == campaign_id]
    except Exception as e:
        logger.error(f"Failed to fetch campaign history: {e}")
        return pd.DataFrame()


def fetch_variant_performance(campaign_id: str) -> Dict[str, Any]:
    df = fetch_campaign_history(campaign_id)
    if df.empty:
        return {}

    for c in ["conv_rate", "ctr"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df_sorted = df.sort_values(["conv_rate", "ctr"], ascending=False)
    best = df_sorted.iloc[0].to_dict()
    worst = df_sorted.iloc[-1].to_dict()
    return {
        "best": best,
        "worst": worst,
        "all_variants": df_sorted.to_dict(orient="records")
    }


def get_ml_training_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(HISTORICAL_CSV)
        df = df.dropna()
        return df
    except Exception as e:
        logger.error(f"Failed to load ML training data: {e}")
        return pd.DataFrame()


def build_feature_vector(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ctr": float(row.get("ctr", 0.0)),
        "sentiment": float(row.get("sentiment", 0.0)),
        "polarity": float(row.get("polarity", row.get("sentiment", 0.0))),
        "trend_score": float(row.get("trend_score", 0.0)),
        "conversions": int(row.get("conversions", 0))
    }


def compute_variant_score(row: Dict[str, Any]) -> float:
    try:
        return round(
            float(row.get("ctr", 0.0)) * 0.5 +
            float(row.get("sentiment", 0.0)) * 0.3 +
            float(row.get("trend_score", 0.0)) * 0.2,
            4
        )
    except Exception:
        return 0.0
