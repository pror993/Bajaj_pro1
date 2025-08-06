import re
from datetime import datetime
from typing import Tuple, Any


class DateExtractor:
    def __init__(self):
        """Initialize DateExtractor with date patterns."""
        # Date patterns
        self.date_patterns = [
            # DD/MM/YYYY or MM/DD/YYYY
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            # YYYY-MM-DD
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            # DD Month YYYY
            r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
            # Month DD, YYYY
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})',
            # Today, yesterday, tomorrow
            r'\b(today|yesterday|tomorrow)\b',
            # Relative dates
            r'(\d+)\s+(day|week|month|year)s?\s+(ago|from\s+now)',
        ]
        
        self.month_map = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
    
    def extract(self, text: str) -> Tuple[Any, float]:
        """
        Extract date from text.
        
        Args:
            text: Input text to extract date from
            
        Returns:
            Tuple of (date_value, confidence_score)
        """
        text_lower = text.lower()
        
        for pattern in self.date_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    if 'today' in match.group(0):
                        return datetime.now().strftime('%Y-%m-%d'), 0.9
                    elif 'yesterday' in match.group(0):
                        yesterday = datetime.now().replace(day=datetime.now().day - 1)
                        return yesterday.strftime('%Y-%m-%d'), 0.9
                    elif 'tomorrow' in match.group(0):
                        tomorrow = datetime.now().replace(day=datetime.now().day + 1)
                        return tomorrow.strftime('%Y-%m-%d'), 0.9
                    elif len(match.groups()) == 3:
                        # Standard date format
                        groups = match.groups()
                        if groups[0].isdigit() and groups[1].isdigit() and groups[2].isdigit():
                            # DD/MM/YYYY or MM/DD/YYYY format
                            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                            if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                                return f"{year:04d}-{month:02d}-{day:02d}", 0.9
                        elif groups[0].isdigit() and groups[1].lower() in self.month_map and groups[2].isdigit():
                            # DD Month YYYY format
                            day, month_name, year = int(groups[0]), groups[1].lower(), int(groups[2])
                            month = self.month_map[month_name]
                            if 1 <= day <= 31 and 1900 <= year <= 2100:
                                return f"{year:04d}-{month:02d}-{day:02d}", 0.9
                        elif groups[0].lower() in self.month_map and groups[1].isdigit() and groups[2].isdigit():
                            # Month DD, YYYY format
                            month_name, day, year = groups[0].lower(), int(groups[1]), int(groups[2])
                            month = self.month_map[month_name]
                            if 1 <= day <= 31 and 1900 <= year <= 2100:
                                return f"{year:04d}-{month:02d}-{day:02d}", 0.9
                except (ValueError, IndexError):
                    continue
        
        # No date found
        return None, 0.0 