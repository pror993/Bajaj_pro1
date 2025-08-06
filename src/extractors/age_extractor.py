import re
import spacy
from typing import Tuple, Any


class AgeExtractor:
    def __init__(self):
        """Initialize AgeExtractor with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if spaCy model not available
            self.nlp = None
        
        # Regex patterns for age extraction
        self.age_patterns = [
            r'(\d+)\s*(?:year|yr)s?\s*old',
            r'age[:\s]*(\d+)',
            r'(\d+)\s*(?:year|yr)s?\s*of\s*age',
            r'(\d+)\s*(?:year|yr)s?',
        ]
    
    def extract(self, text: str) -> Tuple[Any, float]:
        """
        Extract age from text.
        
        Args:
            text: Input text to extract age from
            
        Returns:
            Tuple of (age_value, confidence_score)
        """
        text_lower = text.lower()
        
        # Try regex patterns first
        for pattern in self.age_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                age_value = int(match.group(1))
                if 0 <= age_value <= 120:  # Reasonable age range
                    return age_value, 0.9
        
        # Fallback to spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "CARDINAL":
                    try:
                        age_value = int(ent.text)
                        if 0 <= age_value <= 120:
                            # Check if it's in age context
                            context_words = ["age", "year", "old", "born"]
                            if any(word in text_lower for word in context_words):
                                return age_value, 0.7
                    except ValueError:
                        continue
        
        # No age found
        return None, 0.0 