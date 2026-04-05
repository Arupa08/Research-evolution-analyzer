import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration management"""
    
    # API Configuration
    SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
    ARXIV_API_URL = "https://export.arxiv.org/api/query"
    GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", None)
    
    # Cache Configuration
    CACHE_DIR = Path("./cache")
    API_CACHE_DIR = CACHE_DIR / "api_responses"
    EMBEDDINGS_CACHE_DIR = CACHE_DIR / "embeddings"
    TOPIC_MODELS_DIR = CACHE_DIR / "topic_models"
    LOGS_DIR = Path("./logs")
    
    # Create directories if they don't exist
    for dir_path in [API_CACHE_DIR, EMBEDDINGS_CACHE_DIR, TOPIC_MODELS_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # API Configuration
    SEMANTIC_SCHOLAR_TIMEOUT = 30
    MAX_PAPERS_PER_REQUEST = 100
    RATE_LIMIT_DELAY = 1  # seconds between requests
    MAX_RETRIES = 3
    BACKOFF_FACTOR = 2
    
    # Topic Modeling Configuration
    MIN_DF = 2  # Minimum document frequency
    MAX_DF = 0.95  # Maximum document frequency
    NUM_TOPICS = "auto"  # BERTopic auto-selects
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast
    
    # Vector Store Configuration
    VECTOR_DIM = 384  # all-MiniLM-L6-v2 dimension
    FAISS_INDEX_TYPE = "Flat"  # Simple exact search
    
    # Data Validation Configuration
    MIN_ABSTRACT_LENGTH = 50  # Minimum characters
    REQUIRED_FIELDS = ["paperId", "title", "year", "authors", "citationCount"]
    
    # Temporal Analysis Configuration
    TIME_WINDOW_YEARS = 3  # For trend detection
    EMERGING_THRESHOLD = 0.7  # Recent + increasing
    DECLINING_THRESHOLD = 0.3  # Recent decrease
    
    # LLM Configuration
    LLM_MODEL = "gemini-2.5-flash"
    LLM_MAX_TOKENS = 500
    LLM_TEMPERATURE = 0.3  # Lower for consistency
    
    @classmethod
    def get_cache_path(cls, author_id: str, suffix: str) -> Path:
        """Generate cache file path for author data"""
        return cls.API_CACHE_DIR / f"{author_id}_{suffix}.json"
    
    @classmethod
    def is_cache_fresh(cls, cache_path: Path, max_age_days: int = 7) -> bool:
        """Check if cached data is fresh (not older than max_age_days)"""
        if not cache_path.exists():
            return False
        
        import time
        file_age_days = (time.time() - cache_path.stat().st_mtime) / (24 * 3600)
        return file_age_days < max_age_days
    
    @classmethod
    def log_event(cls, event_type: str, message: str, metadata: dict = None):
        """Log events to file and console"""
        import datetime
        
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {}
        }
        
        log_file = cls.LOGS_DIR / f"debug_{datetime.date.today()}.json"
        
        # Append to log file
        existing_logs = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                try:
                    existing_logs = json.load(f)
                except:
                    existing_logs = []
        
        existing_logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(existing_logs, f, indent=2)
        
        print(f"[{event_type}] {message}")
