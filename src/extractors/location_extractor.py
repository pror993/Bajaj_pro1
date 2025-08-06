import spacy
from typing import Tuple, Any


class LocationExtractor:
    def __init__(self):
        """Initialize LocationExtractor with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
    
    def extract(self, text: str) -> Tuple[Any, float]:
        """
        Extract location from text.
        
        Args:
            text: Input text to extract location from
            
        Returns:
            Tuple of (location_value, confidence_score)
        """
        if not self.nlp:
            return None, 0.0
        
        doc = self.nlp(text)
        locations = []
        
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:  # Geographical, Location, Facility
                locations.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        if locations:
            # Return the first location found
            location = locations[0]
            confidence = 0.8 if location['type'] in ["GPE", "LOC"] else 0.6
            return location, confidence
        
        # No location found
        return None, 0.0 