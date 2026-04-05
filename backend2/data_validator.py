import pandas as pd
from typing import List, Dict, Tuple
from config import Config
import json

class DataValidator:
    """Validate and clean fetched paper data with graceful degradation"""
    
    @staticmethod
    def validate_paper(paper: Dict) -> Tuple[bool, str, Dict]:
        # ... (Previous checks for paperId, missing_fields remain the same) ...
        if not paper.get("paperId"):
             return False, "Missing paperId", {}

        # Abstract check
        has_abstract = bool(paper.get("abstract")) and len(paper.get("abstract", "")) >= Config.MIN_ABSTRACT_LENGTH
        
        # Check core metadata
        for field in ["title", "year", "authors", "citationCount"]:
            if field not in paper or paper[field] is None:
                return False, f"Missing field: {field}", {}

        # ... (Year and Author validation remain the same) ...

        # Clean and prepare paper
        # [CHANGE]: Preserve authorId in the authors list for accurate collaboration tracking
        cleaned_authors = []
        for author in paper["authors"]:
            cleaned_authors.append({
                "name": author.get("name", "Unknown"),
                "authorId": author.get("authorId")  # Keep the ID!
            })

        cleaned_paper = {
            "paperId": paper["paperId"],
            "title": paper["title"],
            "year": paper["year"],
            "authors": cleaned_authors, # Updated list with IDs
            "citationCount": paper["citationCount"],
            "abstract": paper.get("abstract", ""),
            "hasAbstract": has_abstract,
            "fieldsOfStudy": paper.get("fieldsOfStudy", []),
            "isOpenAccess": paper.get("isOpenAccess", False),
            "openAccessPdf": paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None,
            "venue": paper.get("venue", "Unknown")
        }
        
        return True, "Valid", cleaned_paper

    # ... (Rest of the class methods validate_papers, papers_to_dataframe etc. remain the same) ...
    # Be sure to include the other methods from the original file
    @staticmethod
    def validate_papers(papers: List[Dict]) -> Tuple[List[Dict], Dict]:
        valid_papers = []
        validation_report = {
            "total_input": len(papers),
            "valid": 0,
            "discarded": 0,
            "with_abstract": 0,
            "without_abstract": 0,
            "discard_reasons": {}
        }
        
        for paper in papers:
            is_valid, reason, cleaned_paper = DataValidator.validate_paper(paper)
            
            if is_valid:
                valid_papers.append(cleaned_paper)
                validation_report["valid"] += 1
                if cleaned_paper["hasAbstract"]:
                    validation_report["with_abstract"] += 1
                else:
                    validation_report["without_abstract"] += 1
            else:
                validation_report["discarded"] += 1
                validation_report["discard_reasons"][reason] = validation_report["discard_reasons"].get(reason, 0) + 1
                
        return valid_papers, validation_report

    @staticmethod
    def papers_to_dataframe(papers: List[Dict]) -> pd.DataFrame:
        if not papers:
            return pd.DataFrame()
        df = pd.DataFrame(papers)
        df["year"] = df["year"].astype(int)
        df["citationCount"] = df["citationCount"].astype(int)
        df["hasAbstract"] = df["hasAbstract"].astype(bool)
        return df

    @staticmethod
    def get_texts_for_topic_modeling(papers: List[Dict], abstract_only: bool = False) -> List[str]:
        texts = []
        for paper in papers:
            if abstract_only and not paper.get("hasAbstract"):
                continue
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            text = f"{title}. {abstract}" if abstract else title
            if len(text.strip()) > Config.MIN_ABSTRACT_LENGTH:
                texts.append(text)
        return texts

    @staticmethod
    def get_metadata_for_vectors(papers: List[Dict]) -> List[Dict]:
        metadata_list = []
        for paper in papers:
            metadata = {
                "paperId": paper["paperId"],
                "title": paper["title"],
                "year": paper["year"],
                "authors": paper["authors"],
                "citationCount": paper["citationCount"],
                "fieldsOfStudy": paper["fieldsOfStudy"],
                "venue": paper["venue"]
            }
            metadata_list.append(metadata)
        return metadata_list

def validate_and_prepare_papers(papers: List[Dict], abstract_only: bool = False):
    valid_papers, report = DataValidator.validate_papers(papers)
    if not valid_papers:
        Config.log_event("VALIDATION_FAILED", "No valid papers after validation")
        return pd.DataFrame(), [], [], report
    
    df = DataValidator.papers_to_dataframe(valid_papers)
    texts = []
    indices_to_keep = []
    
    for idx, row in df.iterrows():
        text = f"{row.get('title', '')}. {row.get('abstract', '')}"
        if len(text.strip()) > Config.MIN_ABSTRACT_LENGTH:
            texts.append(text)
            indices_to_keep.append(idx)
            
    df_filtered = df.loc[indices_to_keep].reset_index(drop=True)
    metadata = DataValidator.get_metadata_for_vectors(df_filtered.to_dict('records'))
    
    return df_filtered, texts, metadata, report