import os
import time
import joblib
import logging
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# Integrations
from app.integrations.sheets_connector import read_rows, append_row
from app.integrations.slack_notifier import SlackNotifier

# AI Components
from app.sentiment_engine.sentiment_analyzer import analyze_sentiment
from app.integrations.trend_fetcher import TrendFetcher
from app.ml_engine.train_model import train  # Your existing training function

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


class AutoRetrainer:
    """
    Automatically retrains your ML model using:
      - A/B test results (labels)
      - Sentiment scores
      - Trend scores
      - Engagement metrics (dummy fallback for now)
    """

    def __init__(self):
        self.sentiment = analyze_sentiment
        self.trends = TrendFetcher()
        self.slack = SlackNotifier()

        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

        # Scheduler for periodic auto-training
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

        logger.info("AutoRetrainer initialized (No SocialIngestor).")

    # -------------------------------------------------------------------
    # 1. LOAD TRAINING DATA FROM GOOGLE SHEETS
    # -------------------------------------------------------------------
    def load_training_data(self) -> pd.DataFrame:
        try:
            rows = read_rows("ab_test_results")
            if not rows or len(rows) < 2:
                logger.warning("No A/B training data available in Sheets.")
                return pd.DataFrame()

            df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows[0]) > 0 else pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"Error loading Sheets data: {e}")
            return pd.DataFrame()

        required_cols = {"postA", "scoreA", "postB", "scoreB", "winner"}
        if not required_cols.issubset(df.columns):
            logger.warning("A/B data missing required columns.")
            return pd.DataFrame()

        return df

    # -------------------------------------------------------------------
    # 2. PREPROCESSING PIPELINE
    # -------------------------------------------------------------------
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        logger.info("Preprocessing training data...")

        processed_rows = []

        for _, row in df.iterrows():
            postA = row.get("postA")
            postB = row.get("postB")
            winner = row.get("winner")

            
            textA = str(postA)
            textB = str(postB)

            # Sentiment
            sentA = analyze_sentiment(textA)[0].get("sentiment_score", 0)
            sentB = analyze_sentiment(textB)[0].get("sentiment_score", 0)

            # Trend score
            trendA = self.trends.get_combined_trend_score(textA)
            trendB = self.trends.get_combined_trend_score(textB)

            # Dummy engagement (until replaced with real metrics source)
            engA = 0
            engB = 0

            # Label = 1 if A wins else 0
            label = 1 if str(winner).lower() == "a" else 0

            processed_rows.append({
                "textA": textA,
                "textB": textB,
                "sentA": sentA,
                "sentB": sentB,
                "trendA": trendA,
                "trendB": trendB,
                "engA": engA,
                "engB": engB,
                "label": label
            })

        return pd.DataFrame(processed_rows)

    # -------------------------------------------------------------------
    # 3. MINIMUM DATA CHECK
    # -------------------------------------------------------------------
    def check_threshold(self, df: pd.DataFrame) -> bool:
        threshold = 20
        if len(df) >= threshold:
            logger.info(f"Enough training samples ({len(df)}). Ready to retrain.")
            return True
        logger.info(f"Not enough data for retraining: {len(df)}/{threshold}")
        return False

    # -------------------------------------------------------------------
    # 4. RETRAIN MODEL
    # -------------------------------------------------------------------
    def retrain(self, df: pd.DataFrame):
        if df.empty:
            logger.warning("Cannot retrain — empty dataset.")
            return None

        logger.info("Retraining ML model...")

        X = df[["sentA", "sentB", "trendA", "trendB", "engA", "engB"]]
        y = df["label"]

        model = train(X, y)
        return model

    # -------------------------------------------------------------------
    # 5. SAVE MODEL VERSION
    # -------------------------------------------------------------------
    def save_model(self, model, df: pd.DataFrame):
        timestamp = int(time.time())
        path = os.path.join(self.model_dir, f"model_{timestamp}.pkl")
        joblib.dump(model, path)

        try:
            append_row("model_versions", [timestamp, len(df), path])
        except Exception:
            pass

        logger.info(f"New model saved: {path}")
        return path

    # -------------------------------------------------------------------
    # 6. SLACK NOTIFICATION
    # -------------------------------------------------------------------
    def notify_slack(self, path: str, dfsize: int):
        if not self.slack:
            return
        try:
            self.slack.send_message(f"📈 New ML Model Retrained!\nSize: {dfsize} samples\nSaved: {path}")
        except Exception:
            pass

    # -------------------------------------------------------------------
    # 7. FULL PIPELINE EXECUTION
    # -------------------------------------------------------------------
    def run_full_cycle(self):
        logger.info("Starting retraining cycle...")

        df_raw = self.load_training_data()
        if df_raw.empty:
            return

        df = self.preprocess_data(df_raw)
        if not self.check_threshold(df):
            return

        model = self.retrain(df)
        if not model:
            return

        path = self.save_model(model, df)
        self.notify_slack(path, len(df))

        logger.info("Retraining cycle complete.")

    def run(self):
        try:
            logger.info("AutoRetrainer.run() invoked.")
            self.run_full_cycle()
        except Exception as e:
            logger.error(f"AutoRetrainer.run() failed: {e}")
            raise

    # -------------------------------------------------------------------
    # 8. AUTOMATIC SCHEDULING
    # -------------------------------------------------------------------
    def schedule_retraining(self, interval_hours: int = 24):
        self.scheduler.add_job(self.run_full_cycle, 'interval', hours=interval_hours)
        logger.info(f"Auto-retraining scheduled every {interval_hours} hours.")



