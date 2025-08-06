import re
from typing import Tuple, Any


class VehicleExtractor:
    def __init__(self):
        """Initialize VehicleExtractor with vehicle type patterns."""
        # Vehicle type patterns
        self.vehicle_patterns = {
            'car': [
                r'\b(?:car|sedan|hatchback|coupe|suv|truck|van)\b',
                r'\b(?:automobile|vehicle|motor\s*vehicle)\b',
                r'\b(?:honda|toyota|ford|bmw|mercedes|audi|volkswagen)\b'
            ],
            'motorcycle': [
                r'\b(?:motorcycle|bike|scooter|moped)\b',
                r'\b(?:harley|yamaha|kawasaki|ducati)\b'
            ],
            'commercial': [
                r'\b(?:truck|lorry|van|bus|tractor)\b',
                r'\b(?:commercial\s*vehicle|delivery\s*vehicle)\b'
            ],
            'bicycle': [
                r'\b(?:bicycle|bike|cycle)\b',
                r'\b(?:pedal\s*bike|push\s*bike)\b'
            ]
        }
    
    def extract(self, text: str) -> Tuple[Any, float]:
        """
        Extract vehicle type from text.
        
        Args:
            text: Input text to extract vehicle type from
            
        Returns:
            Tuple of (vehicle_type, confidence_score)
        """
        text_lower = text.lower()
        
        # Count matches for each vehicle type
        vehicle_scores = {}
        
        for vehicle_type, patterns in self.vehicle_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                score += len(matches)
            vehicle_scores[vehicle_type] = score
        
        # Find vehicle type with highest score
        if vehicle_scores:
            best_type = max(vehicle_scores, key=vehicle_scores.get)
            if vehicle_scores[best_type] > 0:
                confidence = min(0.9, 0.5 + (vehicle_scores[best_type] * 0.1))
                return best_type, confidence
        
        # No vehicle type found
        return None, 0.0 