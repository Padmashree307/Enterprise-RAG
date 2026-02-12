from typing import List, Optional
from src.config.settings import settings
from src.utils.logger import logger

class QueryProcessor:
    def __init__(self):
        # Define keywords for simple routing
        self.keywords = {
            "finance": ["finance", "accounting", "budget", "expenditure", "cost", "invoice", "transaction", "amount", "euro", "fund", "money", "pay", "bank"],
            "hr": ["hr", "human resource", "employee", "staff", "recruitment", "salary", "benefits", "leave", "contract", "personnel", "hiring", "job"],
            "manufacturing": ["manufacturing", "production", "factory", "plant", "machine", "equipment", "maintenance", "output", "industry", "sector", "region"]
        }

    def detect_departments(self, query: str) -> List[str]:
        """
        Detects relevant departments based on query keywords.
        Returns a list of department keys (e.g. ['finance', 'hr']).
        If no keywords match, returns all departments (fallback).
        """
        query_lower = query.lower()
        detected = []
        
        for dept, keys in self.keywords.items():
            if any(k in query_lower for k in keys):
                detected.append(dept)
                
        if not detected:
            logger.info("No department keywords detected. Searching all.")
            return list(self.keywords.keys())
            
        logger.info(f"Detected departments: {detected}")
        return detected

    def preprocess_query(self, query: str) -> str:
        """
        Cleans and normalizes the query.
        """
        return query.strip()

query_processor = QueryProcessor()
