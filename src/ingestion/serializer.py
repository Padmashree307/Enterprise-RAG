from typing import Dict, Any

def serialize_record(record: Dict[str, str], department: str) -> str:
    """
    Converts a structured record dictionary into a natural language string.
    
    Args:
        record (Dict[str, str]): Key-value pairs of the record.
        department (str): Department name (lowercase).
        
    Returns:
        str: Natural language representation.
    """
    # Department-specific templates could be added here
    # For now, a generic key-value to sentence conversion is robust
    
    parts = []
    
    # Handle Department mismatch or redundancy
    # The record usually starts with "Department: ..." which we can skip or use
    
    for key, value in record.items():
        # Clean keys (underscores to spaces)
        clean_key = key.replace("_", " ").title()
        
        # Skip empty or "Not Available" if we want to be concise? 
        # Requirement says: HR "Not Available" is intentional. Keep it.
        
        parts.append(f"{clean_key} is {value}")
        
    # Join with comma or period
    text = ". ".join(parts) + "."
    return text
