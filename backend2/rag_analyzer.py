import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
from config import Config
from scipy.stats import linregress

class RAGAnalyzer:
    """Temporal RAG analysis layer."""
    
    def __init__(self, papers_df: pd.DataFrame, vector_store=None, topic_modeler=None):
        self.papers_df = papers_df
        self.vector_store = vector_store
        self.topic_modeler = topic_modeler

    def analyze_collaborations(self, target_author_id: str) -> Dict:
        """
        [IMP #4] Fix & Enrich Collaboration Intelligence
        """
        if self.papers_df.empty or 'authors' not in self.papers_df.columns:
            return {"collaborations": []}
        
        collaborations = {}
        total_papers = len(self.papers_df)
        current_year = 2025 # Or datetime.now().year
        
        for idx, row in self.papers_df.iterrows():
            authors = row.get('authors', [])
            year = int(row.get('year', 0))
            topic = row.get('topic_id', -1)
            
            paper_collaborators = []
            if isinstance(authors, list):
                for auth in authors:
                    if isinstance(auth, dict):
                        a_id = str(auth.get('authorId', ''))
                        a_name = auth.get('name', 'Unknown')
                        if a_id == str(target_author_id): continue
                        paper_collaborators.append(a_name)
                    elif isinstance(auth, str):
                        paper_collaborators.append(auth)

            paper_collaborators = sorted(list(set(paper_collaborators)))

            for collaborator in paper_collaborators:
                key = collaborator
                if key not in collaborations:
                    collaborations[key] = {
                        "author1": "Target",
                        "author2": collaborator,
                        "papers_together": 0,
                        "years_active": set(),
                        "topics": set()
                    }
                collaborations[key]["papers_together"] += 1
                if year > 0: collaborations[key]["years_active"].add(year)
                if topic != -1: collaborations[key]["topics"].add(int(topic))

        # Format and Enrich
        formatted_collabs = []
        for collab in collaborations.values():
            count = collab["papers_together"]
            years = sorted(list(collab["years_active"]))
            last_year = years[-1] if years else 0
            
            # [IMP #4] Role Classification
            role = "Occasional"
            if count >= max(3, total_papers * 0.1): # >10% of papers or >3 papers
                role = "Core Collaborator"
            elif last_year < (current_year - 3):
                role = "Past Collaborator"
            
            # Strength Score (0-100)
            strength = min(100, (count / max(1, total_papers)) * 100 * 1.5) 

            formatted_collabs.append({
                "author1": collab["author2"], # The collaborator name
                "author2": "Target Author",
                "papers_together": count,
                "years_active": years,
                "topics": sorted(list(collab["topics"])),
                "role": role,           # New Field
                "strength": int(strength) # New Field
            })
            
        sorted_collabs = sorted(formatted_collabs, key=lambda x: x['papers_together'], reverse=True)
        return {"collaborations": sorted_collabs}

    def predict_future_directions(self, model_stats: Dict, evolution: Dict) -> Dict:
        """
        [IMP #2] Predict using real trend signals (slope)
        """
        predictions = []
        if not model_stats.get("topics"): return {"predictions": []}
        
        evolution_data = evolution.get("evolution_by_year", {})
        if not evolution_data: return {"predictions": []}
        
        years = sorted([int(y) for y in evolution_data.keys()])
        if len(years) < 3: return {"predictions": []}
        
        topics = model_stats.get("topics", [])
        
        for topic in topics:
            topic_id = str(topic['id'])
            
            # Create time series
            counts = [evolution_data.get(y, {}).get(topic_id, 0) for y in years]
            
            # Linear Regression for Slope
            if sum(counts) > 2:
                slope, _, _, _, _ = linregress(years, counts)
            else:
                slope = 0
            
            # Filter low confidence
            if slope <= 0.1: continue
            
            recent_volume = sum(counts[-2:])
            
            # Normalize slope for display (0.0 - 1.0)
            momentum_score = min(1.0, slope / 2.0)
            
            direction_text = " + ".join(topic['keywords'][:3])
            
            predictions.append({
                "direction": direction_text,
                "momentum": float(slope),
                "confidence": float(momentum_score), # Normalized confidence
                "recent_papers": int(recent_volume),
                "trend_type": "Accelerating" if slope > 0.5 else "Growing"
            })
            
        predictions.sort(key=lambda x: x['momentum'], reverse=True)
        return {"predictions": predictions[:5]}

def analyze_author_intelligence(papers_df, model_stats, evolution, target_author_id):
    analyzer = RAGAnalyzer(papers_df)
    collaborations = analyzer.analyze_collaborations(target_author_id)
    predictions = analyzer.predict_future_directions(model_stats, evolution)
    return collaborations, predictions