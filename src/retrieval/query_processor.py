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

    def extract_record_id(self, query: str) -> Optional[str]:
        """
        Extracts a specific record ID from the query using regex.
        Example: "tell me about FIN-001" -> "FIN-001"
        """
        match = self.id_pattern.search(query)
        if match:
            extracted_id = match.group(0).upper()
            logger.info(f"Extracted record ID from query: {extracted_id}")
            return extracted_id
        return None

    def preprocess_query(self, query: str) -> str:
        """
        Cleans and normalizes the query.
        """
        return query.strip()

query_processor = QueryProcessor()
