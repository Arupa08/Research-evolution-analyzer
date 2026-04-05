import requests
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from config import Config

class SemanticScholarFetcher:
    """Fetch data from Semantic Scholar Official API with caching"""
    
    def __init__(self):
        self.api_url = Config.SEMANTIC_SCHOLAR_API_URL
        self.timeout = Config.SEMANTIC_SCHOLAR_TIMEOUT
        self.cache_dir = Config.API_CACHE_DIR
    
    def resolve_author_id(self, author_name_or_id: str) -> Optional[str]:
        """
        Resolve author name to Semantic Scholar ID.
        If input is already an ID (numeric), return as-is.
        """
        # Check if it's already an ID (numeric)
        if author_name_or_id.isdigit() or author_name_or_id.startswith('corpus-id:'):
            Config.log_event("ID_RESOLUTION", f"Using provided ID: {author_name_or_id}")
            return author_name_or_id
        
        # Search for author by name
        search_url = f"{self.api_url}/author/search"
        params = {"query": author_name_or_id, "limit": 10}
        
        try:
            response = requests.get(search_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            results = response.json()
            
            if results.get("data") and len(results["data"]) > 0:
                author_id = results["data"][0]["authorId"]
                author_name = results["data"][0].get("name", "Unknown")
                Config.log_event("AUTHOR_FOUND", f"Resolved '{author_name_or_id}' to ID: {author_id}")
                return author_id
            else:
                Config.log_event("AUTHOR_NOT_FOUND", f"No author found for: {author_name_or_id}")
                return None
        
        except requests.exceptions.RequestException as e:
            Config.log_event("API_ERROR", f"Failed to resolve author: {str(e)}")
            return None
    
    def fetch_author_papers(
        self,
        author_id: str,
        limit: int = 50,
        start_year: int = 1990,
        end_year: int = 2025
    ) -> List[Dict]:
        """
        Fetch papers for an author using official Semantic Scholar API.
        Uses local cache to avoid repeated API calls.
        """
        
        # Check cache first
        cache_key = f"{author_id}_papers"
        cache_path = Config.get_cache_path(author_id, "papers")
        
        if Config.is_cache_fresh(cache_path, max_age_days=7):
            Config.log_event("CACHE_HIT", f"Using cached papers for author {author_id}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        # Fetch from API
        url = f"{self.api_url}/author/{author_id}/papers"
        
        # Semantic Scholar API fields
        fields = [
            "paperId", "title", "abstract", "year", "authors",
            "citationCount", "referenceCount", "fieldsOfStudy",
            "isOpenAccess", "openAccessPdf", "venue", "publicationDate"
        ]
        
        params = {
            "limit": min(limit, 1000),  # API max is usually 1000
            "fields": ",".join(fields)
        }
        
        papers = []
        retry_count = 0
        
        try:
            while retry_count < Config.MAX_RETRIES:
                try:
                    response = requests.get(
                        url,
                        params=params,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Handle paginated results
                    if "data" in data:
                        papers.extend(data["data"])
                    
                    Config.log_event(
                        "PAPERS_FETCHED",
                        f"Fetched {len(papers)} papers for author {author_id}",
                        {"limit": limit, "start_year": start_year, "end_year": end_year}
                    )
                    
                    break  # Success, exit retry loop
                
                except requests.exceptions.Timeout:
                    retry_count += 1
                    if retry_count < Config.MAX_RETRIES:
                        wait_time = Config.BACKOFF_FACTOR ** retry_count
                        Config.log_event("RETRY", f"Timeout, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
                
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count < Config.MAX_RETRIES:
                        wait_time = Config.BACKOFF_FACTOR ** retry_count
                        Config.log_event("RETRY", f"API error, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise
        
        except Exception as e:
            Config.log_event("FETCH_ERROR", f"Failed to fetch papers: {str(e)}")
            return []
        
        # Filter by year range
        filtered_papers = [
            p for p in papers
            if p.get("year") and start_year <= p["year"] <= end_year
        ]
        
        # Cache results
        with open(cache_path, 'w') as f:
            json.dump(filtered_papers, f, indent=2)
        
        Config.log_event(
            "CACHE_SAVED",
            f"Saved {len(filtered_papers)} papers to cache",
            {"cache_path": str(cache_path)}
        )
        
        return filtered_papers[:limit]
    
    def fetch_author_info(self, author_id: str) -> Optional[Dict]:
        """Fetch author profile information"""
        
        url = f"{self.api_url}/author/{author_id}"
        fields = ["name", "hIndex", "paperCount", "citationCount", "affiliations"]
        
        params = {"fields": ",".join(fields)}
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            Config.log_event("AUTHOR_INFO_ERROR", f"Failed to fetch author info: {str(e)}")
            return None
    
    def fetch_paper_details(self, paper_id: str) -> Optional[Dict]:
        """Fetch full paper details including embedding candidates"""
        
        url = f"{self.api_url}/paper/{paper_id}"
        fields = [
            "paperId", "title", "abstract", "year", "authors",
            "citationCount", "referenceCount", "fieldsOfStudy",
            "isOpenAccess", "openAccessPdf", "venue"
        ]
        
        params = {"fields": ",".join(fields)}
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            Config.log_event("PAPER_DETAIL_ERROR", f"Failed to fetch paper details: {str(e)}")
            return None


# Convenience functions
fetcher = SemanticScholarFetcher()

def fetch_author(author_name_or_id: str, limit: int = 50, start_year: int = 1990, end_year: int = 2025):
    """High-level function to fetch author data"""
    author_id = fetcher.resolve_author_id(author_name_or_id)
    
    if not author_id:
        Config.log_event("FETCH_FAILED", f"Could not resolve author: {author_name_or_id}")
        return None, None
    
    papers = fetcher.fetch_author_papers(author_id, limit, start_year, end_year)
    author_info = fetcher.fetch_author_info(author_id)
    
    return author_info, papers
