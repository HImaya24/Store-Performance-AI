# common/security.py
import re
import html
from typing import Any, Dict, List

class Sanitizer:
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Basic string sanitization"""
        if not value:
            return value
        
        # HTML escape
        value = html.escape(value)
        
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r"<script.*?>.*?</script>",
            r"javascript:",
            r"on\w+=",
            r"vbscript:",
        ]
        
        for pattern in dangerous_patterns:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        
        return value.strip()
    
    @staticmethod
    def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all string values in a dictionary"""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = Sanitizer.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = Sanitizer.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = Sanitizer.sanitize_list(value)
            else:
                sanitized[key] = value
        return sanitized
    
    @staticmethod
    def sanitize_list(data: List[Any]) -> List[Any]:
        """Sanitize all string values in a list"""
        sanitized = []
        for item in data:
            if isinstance(item, str):
                sanitized.append(Sanitizer.sanitize_string(item))
            elif isinstance(item, dict):
                sanitized.append(Sanitizer.sanitize_dict(item))
            elif isinstance(item, list):
                sanitized.append(Sanitizer.sanitize_list(item))
            else:
                sanitized.append(item)
        return sanitized

# Rate limiting storage
REQUEST_COUNTS = {}

def rate_limit(key: str, max_requests: int = 100, window_seconds: int = 3600):
    """Simple rate limiting"""
    import time
    current_time = int(time.time())
    window_key = f"{key}_{current_time // window_seconds}"
    
    if window_key not in REQUEST_COUNTS:
        REQUEST_COUNTS[window_key] = 0
    
    REQUEST_COUNTS[window_key] += 1
    
    # Clean old entries (optional, for memory management)
    old_keys = [k for k in REQUEST_COUNTS.keys() 
                if int(k.split('_')[-1]) < (current_time // window_seconds) - 1]
    for old_key in old_keys:
        del REQUEST_COUNTS[old_key]
    
    return REQUEST_COUNTS[window_key] <= max_requests