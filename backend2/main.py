from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uvicorn
import json
import numpy as np

from config import Config
from data_fetcher import fetch_author
from data_validator import validate_and_prepare_papers
from topic_modeling import analyze_papers
from vector_store import create_vector_store
from rag_analyzer import analyze_author_intelligence
from llm_service import get_llm_service


# -----------------------
# JSON Helpers
# -----------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_to_native(obj):
    if isinstance(obj, dict):
        return {str(k): convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# -----------------------
# FastAPI Setup
# -----------------------
app = FastAPI(
    title="Research Evolution Analyzer",
    version="2.1.0"
)

app.json_encoder = NumpyEncoder
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# API Endpoint
# -----------------------
@app.get("/analyze")
async def analyze(
    author_name_or_id: str = Query(...),
    limit: int = Query(50, ge=10, le=200),
    start_year: int = Query(1990),
    end_year: int = Query(2025),
):
    try:
        # 1️⃣ Fetch Author + Papers
        author_info, papers = fetch_author(
            author_name_or_id, limit, start_year, end_year
        )

        if not author_info or not papers:
            raise HTTPException(status_code=404, detail="Author not found")

        # 2️⃣ Validate + Prepare
        papers_df, texts, metadata, validation_report = validate_and_prepare_papers(papers)

        if papers_df.empty:
            raise HTTPException(status_code=400, detail="No valid papers found")

        # 3️⃣ Topic Modeling
        model_stats, evolution, _, _ = analyze_papers(texts, papers_df)

        # 4️⃣ 🔥 BATCH Topic Labeling (ONE Gemini Call)
        if "topics" in model_stats and model_stats["topics"]:
            llm_service = get_llm_service()

            topic_payload = [
                {
                    "id": t["id"],
                    "keywords": t["keywords"]
                }
                for t in model_stats["topics"]
            ]

            try:
                # ONE request for ALL topics
                labels: Dict[int, str] = llm_service.label_topics_batch(topic_payload)

                for topic in model_stats["topics"]:
                    topic["label"] = labels.get(
                        topic["id"],
                        " / ".join(topic["keywords"][:2])
                    )

            except Exception as e:
                Config.log_event("LLM_WARNING", f"Batch labeling failed: {e}")

                # Safe fallback
                for topic in model_stats["topics"]:
                    topic["label"] = " / ".join(topic["keywords"][:2])

        # 5️⃣ Author Intelligence (Collaborations + Predictions)
        target_id = author_info.get("authorId", author_name_or_id)

        collaborations, predictions = analyze_author_intelligence(
            papers_df,
            model_stats,
            evolution,
            target_id
        )

        # 6️⃣ Vector Store (ONLY if embeddings exist)
        embeddings = model_stats.get("embeddings")
        if isinstance(embeddings, np.ndarray) and embeddings.size > 0:
            create_vector_store(
                embeddings.astype(np.float32),
                papers_df,
                target_id
            )

        # 7️⃣ Response
        response = {
            "author": {
                "name": str(author_info.get("name", "Unknown")),
                "id": str(author_info.get("authorId", "")),
                "total_papers": int(author_info.get("paperCount", len(papers_df))),
                "total_citations": int(author_info.get("citationCount", 0)),
                "h_index": int(author_info.get("hIndex", 0)),
            },
            "topics": [
                {
                    "id": int(t["id"]),
                    "label": str(t.get("label", "Unknown")),
                    "keywords": [str(k) for k in t["keywords"]],
                    "frequency": int(t["frequency"]),
                    "total_citations": int(t.get("total_citations", 0)),
                    "impact_factor": float(t.get("impact_factor", 0.0)),
                }
                for t in model_stats.get("topics", [])
            ],
            "topic_evolution": convert_to_native(
                evolution.get("evolution_by_year", {})
            ),
            "topic_trends": convert_to_native(
                evolution.get("topics", {})
            ),
            "collaborators": [
                {
                    "author1": str(c["author1"]),
                    "author2": str(c["author2"]),
                    "papers_together": int(c["papers_together"]),
                    "years_active": [int(y) for y in c["years_active"]],
                    "role": str(c.get("role", "Occasional")),
                    "strength": int(c.get("strength", 0)),
                    "topics": [str(t) for t in c["topics"]],
                }
                for c in collaborations.get("collaborations", [])[:15]
            ],
            "future_predictions": convert_to_native(
                predictions.get("predictions", [])
            ),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        Config.log_event("API_ERROR", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
