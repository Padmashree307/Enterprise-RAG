from typing import List, Dict, Any
from pathlib import Path
import re
from src.utils.logger import logger
from src.utils.helpers import timer
from src.ingestion.serializer import serialize_record

@timer
def parse_text_file(file_path: Path, department: str) -> List[Dict[str, Any]]:
    """
    Parses a structured text file where records are separated by blank lines
    and fields are key-value pairs (Key: Value. Key2: Value2.)
    
    Args:
        file_path (Path): Path to the text file.
        department (str): Department name.
        
    Returns:
        List[Dict]: List of document dictionaries with serialized text and metadata.
    """
    documents = []
    
    if not file_path.exists():
        logger.error(f"Text file not found: {file_path}")
        return []

    try:
        content = file_path.read_text(encoding="utf-8")
        # Split by double newline to get blocks/records
        # Windows lines might be \r\n, so we normalize
        records_raw = re.split(r'\n\s*\n', content.strip())
        
        logger.info(f"Parsing {len(records_raw)} records from {file_path.name}")
        
        for i, block in enumerate(records_raw):
            if not block.strip():
                continue
                
            # Parse key-values: "Key: Value. Key2: Value2."
            # Use regex to split ONLY on ". " when followed by a field key pattern
            # (capitalized word + colon). This prevents splitting inside decimal
            # numbers like "Amount_EUR: 1475217.88. Status: Approved."
            
            # Remove trailing dot if present
            if block.endswith('.'):
                block = block[:-1]
                
            pairs = re.split(r'\.\s+(?=[A-Z][A-Za-z_]+\s*:)', block)
            record_dict = {}
            
            for pair in pairs:
                if ':' not in pair:
                    continue
                k, v = pair.split(':', 1)
                record_dict[k.strip()] = v.strip()
            
            if not record_dict:
                logger.warning(f"No fields found in block {i} of {file_path.name}")
                continue
            
            # Identify a unique ID for metadata
            # Candidates: "Transaction_ID", "Employee_ID", "Record_ID"
            record_id = (
                record_dict.get("Transaction_ID") or 
                record_dict.get("Transaction Id") or 
                record_dict.get("Employee_ID") or 
                record_dict.get("Employee Id") or 
                record_dict.get("Record_ID") or 
                record_dict.get("Record Id") or 
                f"REC-{i:03d}"
            )
            
            # Serialize
            text_content = serialize_record(record_dict, department)
            
            # Create doc
            doc = {
                "text": text_content,
                "metadata": {
                    "department": department,
                    "source_type": "structured",
                    "source_file": file_path.name,
                    "record_id": record_id
                }
            }
            documents.append(doc)
            
        logger.info(f"Successfully parsed {len(documents)} records from {file_path.name}")
        return documents

    except Exception as e:
        logger.error(f"Failed to parse text file {file_path}: {e}")
        return []
