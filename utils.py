# utils.py
import re

def extract_field_from_text(text_block, field_label):

    if not isinstance(text_block, str) or not isinstance(field_label, str):
        return "N/A"
    
    pattern = rf"(?i){re.escape(field_label)}\s*:\s*(.*?)(?=\n\s*[A-ZÀ-Úa-zÀ-ÖØ-öø-ÿ][\w\sÀ-ÖØ-öø-ÿ()]*\s*:|\Z)"
    match = re.search(pattern, text_block, re.DOTALL)
    
    if match:
        value = match.group(1).strip()
        if value.startswith(".."): value = value[2:].strip()
        elif value.startswith("."): value = value[1:].strip()
        return value if value else "N/A"
    return "N/A"

