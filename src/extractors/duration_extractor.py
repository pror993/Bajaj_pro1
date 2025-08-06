import re
from typing import Tuple, Any


class DurationExtractor:
    def __init__(self):
        """Initialize DurationExtractor with duration patterns."""
        # Regex patterns for duration extraction
        self.duration_patterns = [
            r'(\d+)\s*(?:month|mo)s?',
            r'(\d+)\s*(?:day|d)s?',
            r'(\d+)\s*(?:year|yr)s?',
            r'(\d+)\s*(?:week|wk)s?',
            r'(\d+)\s*(?:hour|hr)s?',
            r'(\d+)\s*(?:minute|min)s?',
        ]
    
    def extract(self, text: str) -> Tuple[Any, float]:
        """
        Extract duration from text.
        
        Args:
            text: Input text to extract duration from
            
        Returns:
            Tuple of (duration_value, confidence_score)
        """
        text_lower = text.lower()
        
        for pattern in self.duration_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                unit = match.group(0).split()[1] if ' ' in match.group(0) else match.group(0)[-2:]
                
                # Normalize to days for comparison
                duration_days = self._normalize_to_days(value, unit)
                
                return {
                    'value': value,
                    'unit': unit,
                    'days': duration_days,
                    'text': match.group(0)
                }, 0.9
        
        # No duration found
        return None, 0.0
    
    def _normalize_to_days(self, value: int, unit: str) -> int:
        """Convert duration to days for comparison."""
        unit_lower = unit.lower()
        
        if 'year' in unit_lower or 'yr' in unit_lower:
            return value * 365
        elif 'month' in unit_lower or 'mo' in unit_lower:
            return value * 30
        elif 'week' in unit_lower or 'wk' in unit_lower:
            return value * 7
        elif 'day' in unit_lower or 'd' in unit_lower:
            return value
        elif 'hour' in unit_lower or 'hr' in unit_lower:
            return value // 24
        elif 'minute' in unit_lower or 'min' in unit_lower:
            return value // (24 * 60)
        else:
            return value 