# 🔬 Research Evolution Analyzer
### Temporal Intelligence Platform for Academic Research Trajectory Analysis

> Automatically converts a researcher's publication history into structured, data-driven research insights using BERTopic, FAISS, and Google Gemini.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![BERTopic](https://img.shields.io/badge/BERTopic-0.15-blue?style=flat)](https://maartengr.github.io/BERTopic)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat&logo=google)](https://aistudio.google.com)

---

## 📌 What It Does

Takes a researcher's name or Semantic Scholar ID and returns a full analysis of their academic trajectory:

- What topics they research and how important each is
- How their research focus has shifted year by year
- Which topics are emerging, stable, or declining
- Who their key collaborators are
- What directions they are likely to pursue next

---

## 🎯 Live Demo

| | Link |
|---|---|
| 🌐 Frontend | `https://YOURUSERNAME.github.io/research-evolution-analyzer/index.html` |
| ⚙️ API Docs | `https://your-app.onrender.com/docs` |

> ⚠️ First load may take 30s on free tier (Render cold start)

---

## 🏗️ Pipeline

```
Researcher Name or Semantic Scholar ID
              │
              ▼
┌──────────────────────────────────┐
│  data_fetcher.py                 │
│  Semantic Scholar Graph API      │
│  • Author ID resolution          │
│  • Fetch papers + metadata       │
│  • 7-day cache + exponential     │
│    backoff retry logic           │
└─────────────┬────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  data_validator.py               │
│  • Remove incomplete papers      │
│  • Enforce required fields       │
│  • Preserve authorId for         │
│    collaboration tracking        │
│  • Build text corpus             │
└─────────────┬────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  topic_modeling.py               │
│  SentenceTransformer             │
│  (all-MiniLM-L6-v2, 384-dim)     │
│  → BERTopic clustering           │
│  → Citation-weighted ranking     │
│  → Linear regression trends      │
└─────────────┬────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  vector_store.py                 │
│  FAISS IndexFlatL2               │
│  • Stores paper embeddings       │
│  • Filter by year + citations    │
│  • Persisted to disk per author  │
└─────────────┬────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  llm_service.py                  │
│  Google Gemini 2.5 Flash         │
│  • Batch topic labeling          │
│    (ONE API call for all topics) │
│  • Keyword fallback if no key    │
└─────────────┬────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  rag_analyzer.py                 │
│  • Co-author graph extraction    │
│  • Role classification           │
│    (Core / Occasional / Past)    │
│  • Momentum-based predictions    │
│    using linregress slope        │
└─────────────┬────────────────────┘
              │
              ▼
       FastAPI /analyze
              │
              ▼
     index.html + Chart.js
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Data Source | Semantic Scholar Graph API |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` |
| Topic Modeling | BERTopic + CountVectorizer |
| Vector Store | FAISS `IndexFlatL2` |
| Trend Analysis | SciPy `linregress` |
| LLM | Google Gemini 2.5 Flash |
| Frontend | HTML + Chart.js + D3.js |

---

## 📁 Project Structure

```
research-evolution-analyzer/
├── main.py              # FastAPI app + /analyze endpoint
├── config.py            # Config, caching, logging
├── data_fetcher.py      # Semantic Scholar API client
├── data_validator.py    # Paper cleaning + preprocessing
├── topic_modeling.py    # BERTopic + trend classification
├── vector_store.py      # FAISS index management
├── rag_analyzer.py      # Collaborations + predictions
├── llm_service.py       # Gemini batch labeling
├── index.html           # Frontend
├── requirements.txt
└── .env                 # API keys (never committed)
```

---

## ⚙️ Running Locally

```bash
# 1. Clone
git clone https://github.com/YOURUSERNAME/research-evolution-analyzer.git
cd research-evolution-analyzer

# 2. Install
pip install -r requirements.txt

# 3. Add API key
echo "GOOGLE_GEMINI_API_KEY=your_key_here" > .env

# 4. Start backend (Terminal 1)
uvicorn main:app --reload

# 5. Serve frontend (Terminal 2)
python -m http.server 3000
```

Open `http://localhost:3000/index.html`

> First run downloads the embedding model (~90MB). Takes 20–30 seconds.

---

## 📡 API

### `GET /analyze`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `author_name_or_id` | string | required | Name or Semantic Scholar numeric ID |
| `limit` | int | 50 | Max papers (10–200) |
| `start_year` | int | 1990 | Filter from year |
| `end_year` | int | 2025 | Filter to year |

**Example:**
```
GET /analyze?author_name_or_id=1688882&limit=50&start_year=2015&end_year=2025
```

**Response includes:**
```json
{
  "author": { "name": "...", "h_index": 0, "total_citations": 0 },
  "topics": [{ "id": 0, "label": "...", "keywords": [], "frequency": 0 }],
  "topic_evolution": { "2015": { "0": 3 }, "2016": { "0": 2 } },
  "topic_trends": { "emerging": [], "stable": [], "declining": [] },
  "collaborators": [{ "author1": "...", "papers_together": 0, "role": "Core Collaborator" }],
  "future_predictions": [{ "direction": "...", "momentum": 0.0, "confidence": 0.0 }]
}
```

Full Swagger docs at `/docs`

---

## 🔑 Key Engineering Decisions

**Batch LLM calls** — All topics labeled in one Gemini request instead of one call per topic, reducing API cost and latency significantly.

**Citation-weighted topic importance** — Ranking uses `total_citations + (frequency × 2)` so high-impact topics surface above just frequent ones.

**Normalized regression axis** — Trend detection uses `np.arange(len(years))` instead of raw year integers to avoid scale distortion in slope calculation.

**authorId preservation** — Co-author IDs are kept through validation to enable accurate deduplication in the collaboration graph.

**Graceful LLM fallback** — System runs fully without a Gemini API key, using top keywords as labels automatically.

---

## 📄 License

MIT License
