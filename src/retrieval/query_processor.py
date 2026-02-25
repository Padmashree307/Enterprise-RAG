import re
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
        # Regex for ID patterns: FIN-001, EMP-101, REC-001, etc.
        self.id_pattern = re.compile(r'\b(FIN-\d+|EMP-\d+|REC-\d+|PROD-\d+)\b', re.IGNORECASE)

    def detect_departments(self, query: str) -> List[str]:
        # ... (unchanged)
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

    def extract_record_ids(self, query: str) -> List[str]:
        """
        Extracts all specific record IDs from the query using regex.
        Example: "tell me about FIN-001 and EMP-101" -> ["FIN-001", "EMP-101"]
        """
        matches = self.id_pattern.findall(query)
        if matches:
            # Deduplicate and normalize
            extracted_ids = list(set(m.upper() for m in matches))
            logger.info(f"Extracted record IDs from query: {extracted_ids}")
            return extracted_ids
        return []

    def preprocess_query(self, query: str) -> str:
        """
        Cleans and normalizes the query.
        """
        return query.strip()

query_processor = QueryProcessor()
