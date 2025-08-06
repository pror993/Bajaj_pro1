import re
from typing import Tuple, Any


class ReasonExtractor:
    def __init__(self):
        """Initialize ReasonExtractor with reason keywords."""
        # Reason keywords by category
        self.reason_keywords = {
            'cancellation': [
                'cancel', 'cancellation', 'postpone', 'delay', 'reschedule',
                'unavailable', 'not available', 'changed mind'
            ],
            'medical': [
                'sick', 'illness', 'injury', 'medical', 'health', 'doctor',
                'hospital', 'emergency', 'pain', 'symptoms'
            ],
            'personal': [
                'personal', 'family', 'emergency', 'urgent', 'important',
                'unexpected', 'unforeseen', 'circumstances'
            ],
            'technical': [
                'technical', 'system', 'error', 'fault', 'breakdown',
                'maintenance', 'repair', 'issue', 'problem'
            ],
            'financial': [
                'cost', 'expensive', 'budget', 'money', 'financial',
                'payment', 'afford', 'price'
            ]
        }
    
    def extract(self, text: str) -> Tuple[Any, float]:
        """
        Extract reason from text.
        
        Args:
            text: Input text to extract reason from
            
        Returns:
            Tuple of (reason_value, confidence_score)
        """
        text_lower = text.lower()
        
        # Count matches for each reason category
        reason_scores = {}
        
        for category, keywords in self.reason_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            reason_scores[category] = score
        
        # Find reason category with highest score
        if reason_scores:
            best_category = max(reason_scores, key=reason_scores.get)
            if reason_scores[best_category] > 0:
                confidence = min(0.9, 0.5 + (reason_scores[best_category] * 0.1))
                return {
                    'category': best_category,
                    'text': text,
                    'confidence': confidence
                }, confidence
        
        # No reason found
        return None, 0.0 