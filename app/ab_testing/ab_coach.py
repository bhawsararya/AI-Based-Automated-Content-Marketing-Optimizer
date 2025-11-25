# ab_coach.py

import os
import joblib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from app.integrations.sheets_connector import append_row, read_rows
from app.integrations.slack_notifier import SlackNotifier
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment
from app.integrations.trend_fetcher import TrendFetcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

MODEL_DIR = os.getenv("MODEL_DIR", "models")
SHEETS_ENABLED = bool(os.getenv("GOOGLE_SHEET_ID"))


def _load_latest_model() -> Optional[Any]:
    try:
        if not os.path.exists(MODEL_DIR):
            return None

        files = [
            os.path.join(MODEL_DIR, f)
            for f in os.listdir(MODEL_DIR)
            if f.endswith(".pkl") or f.endswith(".joblib")
        ]

        if not files:
            return None

        latest = max(files, key=os.path.getmtime)
        logger.info(f"Loading model: {latest}")
        return joblib.load(latest)

    except Exception as e:
        logger.warning(f"Model load failed: {e}")
        return None


class ABCoach:

    def __init__(self):
        self.trends = TrendFetcher()
        self.slack = SlackNotifier() if 'SlackNotifier' in globals() else None
        self.model = _load_latest_model()

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat()

    def _persist_schedule(self, ab_id: str, campaign_id: str, jobA: str, jobB: str,
                          runA: str, runB: str, eval_time: str):
        if not SHEETS_ENABLED:
            return
        append_row("ab_schedule",
                   [self._now_iso(), ab_id, campaign_id, jobA, jobB, runA, runB, eval_time])

    def _persist_ab_posts(self, ab_id: str, campaign_id: str, variant: str,
                          post_id: str, ts: Optional[str] = None):
        if not SHEETS_ENABLED:
            return
        append_row("ab_posts",
                   [ts or self._now_iso(), ab_id, campaign_id, variant, post_id])

    def _persist_ab_result(self, ab_id: str, postA: str, scoreA: int,
                           postB: str, scoreB: int, winner: str):
        if not SHEETS_ENABLED:
            return
        append_row("ab_test_results",
                   [self._now_iso(), ab_id, postA, scoreA, postB, scoreB, winner])

    def predict_success(self, text: str) -> float:
        try:
            sentiment_data = analyze_sentiment(text)[0]
            sentiment_score = sentiment_data.get("sentiment_score", 0)
            trend_score = self.trends.get_combined_trend_score(text)
            length_score = min(len(text.split()), 100)

            features = [[sentiment_score, trend_score, length_score]]

            if self.model:
                try:
                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(features)[0]
                        probability = float(proba[-1]) if len(proba) > 1 else float(proba[0])
                    else:
                        pred = self.model.predict(features)[0]
                        probability = 0.75 if pred == 1 else 0.25

                    return max(0.0, min(1.0, probability))
                except Exception:
                    pass

            heuristic = (
                (sentiment_score * 0.5) +
                (trend_score / 100 * 0.4) +
                (length_score / 100 * 0.1)
            )
            return float(max(0.0, min(1.0, heuristic)))

        except Exception:
            return 0.5

    def evaluate_ab_test(self, ab_id: str) -> Optional[Dict[str, Any]]:
        try:
            rows = read_rows("ab_posts") if SHEETS_ENABLED else []
        except:
            rows = []

        postA = postB = campaign_id = None

        for r in rows:
            if len(r) >= 5 and r[1] == ab_id:
                campaign_id = r[2]
                if r[3] == "A":
                    postA = r[4]
                elif r[3] == "B":
                    postB = r[4]

        if not postA or not postB:
            return None

        scoreA = scoreB = 0  # No metrics available now since SocialIngestor removed
        winner = "A" if scoreA > scoreB else "B" if scoreB > scoreA else "tie"

        self._persist_ab_result(ab_id, postA, scoreA, postB, scoreB, winner)

        if self.slack:
            try:
                self.slack.send_message(f"A/B Test {ab_id} winner: {winner}")
            except:
                pass

        return {
            "ab_id": ab_id,
            "campaign_id": campaign_id,
            "postA": postA,
            "postB": postB,
            "scoreA": scoreA,
            "scoreB": scoreB,
            "winner": winner
        }

    def reevaluate_recent(self, lookback_hours: int = 24) -> List[Dict[str, Any]]:
        results = []
        try:
            rows = read_rows("ab_schedule") if SHEETS_ENABLED else []
        except:
            return results

        for r in rows:
            if len(r) >= 8:
                eval_time = datetime.fromisoformat(r[7])
                if eval_time <= datetime.utcnow():
                    res = self.evaluate_ab_test(r[1])
                    if res:
                        results.append(res)

        return results

    def list_scheduled_ab_tests(self) -> List[Dict[str, Any]]:
        try:
            return read_rows("ab_schedule") if SHEETS_ENABLED else []
        except:
            return []

    def simulate_ab(self, textA: str, textB: str):
        scoreA = len(textA) % 100 / 100
        scoreB = len(textB) % 100 / 100
        winner = "A" if scoreA >= scoreB else "B"

        return {
            "scoreA": scoreA,
            "scoreB": scoreB,
            "winner": winner,
            "explanation": f"Variant {winner} seems stronger based on basic scoring."
        }
