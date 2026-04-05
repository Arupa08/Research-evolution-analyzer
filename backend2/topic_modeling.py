import pandas as pd
import numpy as np
from typing import List, Dict
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import linregress
from config import Config


class TemporalTopicModeler:
    """
    BERTopic-based temporal topic modeling with
    citation-aware importance and statistically sane trend detection.
    """

    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.topic_model = None
        self.papers_df = None
        self.texts = None
        self.embeddings = None

        # Strict academic noise removal
        self.vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

    # ------------------------------
    # MODEL FITTING
    # ------------------------------
    def fit_model(self, texts: List[str], papers_df: pd.DataFrame) -> Dict:
        if not texts or len(texts) < 5:
            return {"error": "Not enough documents for topic modeling"}

        self.embeddings = self.embedding_model.encode(
            texts, show_progress_bar=True
        )

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            vectorizer_model=self.vectorizer_model,
            min_topic_size=max(2, len(texts) // 20),
            calculate_probabilities=True,
            verbose=False
        )

        topics, probs = self.topic_model.fit_transform(
            texts, embeddings=self.embeddings
        )

        self.texts = texts
        self.papers_df = papers_df.copy()
        self.papers_df["topic_id"] = topics
        self.papers_df["topic_probability"] = [
            float(np.max(p)) if len(p) else 0.0 for p in probs
        ]

        return self._get_model_stats()

    # ------------------------------
    # TOPIC IMPORTANCE (CITATION AWARE)
    # ------------------------------
    def _get_model_stats(self) -> Dict:
        if self.topic_model is None:
            return {"error": "Model not fitted"}

        topic_info = self.topic_model.get_topic_info()
        valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()

        topics_out = []

        for tid in valid_topics:
            tid = int(tid)
            papers = self.papers_df[self.papers_df["topic_id"] == tid]

            if papers.empty:
                continue

            words = self.topic_model.get_topic(tid) or []
            keywords = [w for w, _ in words[:5]]

            freq = len(papers)
            total_citations = int(papers["citationCount"].sum())
            avg_citations = float(papers["citationCount"].mean())

            # Citation-weighted importance (stable & interpretable)
            importance_score = total_citations + (freq * 2)

            topics_out.append({
                "id": tid,
                "keywords": keywords,
                "frequency": freq,
                "total_citations": total_citations,
                "impact_factor": avg_citations,
                "importance_score": importance_score
            })

        topics_out.sort(key=lambda x: x["importance_score"], reverse=True)

        return {
            "num_topics": len(topics_out),
            "total_documents": len(self.papers_df),
            "topics": topics_out[:20]
        }

    # ------------------------------
    # TEMPORAL EVOLUTION
    # ------------------------------
    def analyze_topic_evolution(self) -> Dict:
        if self.papers_df is None:
            return {"error": "Model not fitted"}

        evolution = {}

        for year in sorted(self.papers_df["year"].unique()):
            year_df = self.papers_df[self.papers_df["year"] == year]
            evolution[int(year)] = {
                str(k): int(v)
                for k, v in year_df["topic_id"].value_counts().items()
            }

        return self._classify_topic_trends(evolution)

    # ------------------------------
    # TREND CLASSIFICATION (FIXED)
    # ------------------------------
    def _classify_topic_trends(self, evolution: Dict) -> Dict:
        result = {
            "evolution_by_year": evolution,
            "topics": {"emerging": [], "stable": [], "declining": []}
        }

        if not evolution or len(evolution) < 3:
            return result

        years = sorted(evolution.keys())
        all_topics = set()

        for yearly in evolution.values():
            all_topics.update(int(k) for k in yearly.keys())

        for topic_id in all_topics:
            if topic_id == -1:
                continue

            counts = [evolution[y].get(str(topic_id), 0) for y in years]
            total_count = sum(counts)

            # HARD GATE: ignore sparse topics
            if total_count < 4:
                continue

            # NORMALIZED TIME AXIS (CRITICAL FIX)
            x = np.arange(len(years))
            slope, _, _, _, _ = linregress(x, counts)

            words = self.topic_model.get_topic(topic_id) or []
            keywords = [w for w, _ in words[:5]]

            topic_info = {
                "id": topic_id,
                "keywords": keywords,
                "total_papers": total_count,
                "trend_slope": float(slope)
            }

            # REALISTIC THRESHOLDS FOR ACADEMIC DATA
            if slope > 0.1:
                result["topics"]["emerging"].append(topic_info)
            elif slope < -0.1:
                result["topics"]["declining"].append(topic_info)
            else:
                result["topics"]["stable"].append(topic_info)

        return result

    # ------------------------------
    # PLACEHOLDERS (COMPATIBILITY)
    # ------------------------------
    def get_collaboration_graph(self):
        return {}

    def predict_future_directions(self):
        return {}


# ------------------------------
# PIPELINE ENTRY
# ------------------------------
def analyze_papers(texts: List[str], papers_df: pd.DataFrame):
    modeler = TemporalTopicModeler()
    model_stats = modeler.fit_model(texts, papers_df)

    if "error" in model_stats:
        return model_stats, {}, {}, {}

    evolution = modeler.analyze_topic_evolution()
    return model_stats, evolution, {}, {}
