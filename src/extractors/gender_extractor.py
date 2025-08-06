import re
from typing import Tuple, Any


class GenderExtractor:
    def __init__(self):
        """Initialize GenderExtractor with gender patterns."""
        # Regex patterns for gender extraction
        self.gender_patterns = {
            'male': [
                r'\b(?:male|m|man|boy|gentleman)\b',
                r'\b(?:he|him|his)\b',
                r'\b(?:mr|mister)\b',
            ],
            'female': [
                r'\b(?:female|f|woman|girl|lady)\b',
                r'\b(?:she|her|hers)\b',
                r'\b(?:mrs|miss|ms)\b',
            ]
        }
    
    def extract(self, text: str) -> Tuple[Any, float]:
        """
        Extract gender from text.
        
        Args:
            text: Input text to extract gender from
            
        Returns:
            Tuple of (gender_value, confidence_score)
        """
        text_lower = text.lower()
        
        # Count matches for each gender
        gender_scores = {}
        
        for gender, patterns in self.gender_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                score += len(matches)
            gender_scores[gender] = score
        
        # Determine gender with highest score
        if gender_scores['male'] > gender_scores['female']:
            return 'male', min(0.9, 0.5 + (gender_scores['male'] * 0.1))
        elif gender_scores['female'] > gender_scores['male']:
            return 'female', min(0.9, 0.5 + (gender_scores['female'] * 0.1))
        else:
            # No clear gender indication
            return None, 0.0 