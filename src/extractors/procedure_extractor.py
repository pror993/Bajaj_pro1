import spacy
from typing import Tuple, Any


class ProcedureExtractor:
    def __init__(self):
        """Initialize ProcedureExtractor with spaCy model and medical keywords."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
        
        # Medical procedure keywords
        self.procedure_keywords = {
            'surgery': ['surgery', 'operation', 'surgical', 'procedure'],
            'diagnostic': ['x-ray', 'mri', 'ct scan', 'ultrasound', 'blood test'],
            'treatment': ['therapy', 'treatment', 'medication', 'prescription'],
            'emergency': ['emergency', 'urgent', 'critical', 'acute'],
            'preventive': ['checkup', 'screening', 'vaccination', 'preventive']
        }
    
    def extract(self, text: str) -> Tuple[Any, float]:
        """
        Extract medical procedure from text.
        
        Args:
            text: Input text to extract procedure from
            
        Returns:
            Tuple of (procedure_value, confidence_score)
        """
        text_lower = text.lower()
        
        # Check for procedure keywords
        found_procedures = []
        
        for category, keywords in self.procedure_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_procedures.append((category, keyword))
        
        if found_procedures:
            # Return the first found procedure with high confidence
            procedure_type, keyword = found_procedures[0]
            return {
                'type': procedure_type,
                'keyword': keyword,
                'text': text
            }, 0.8
        
        # Fallback to spaCy NER
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PROCEDURE", "TREATMENT"]:
                    return {
                        'type': 'detected',
                        'procedure': ent.text,
                        'text': text
                    }, 0.6
        
        # No procedure found
        return None, 0.0 