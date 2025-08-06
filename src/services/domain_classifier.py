from transformers import pipeline
from typing import Tuple


class DomainClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize domain classifier with a pretrained DistilBERT model.
        The model should be fine-tuned on your specific domains.
        """
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True
        )
        
        # Domain mapping - adjust based on your model's output classes
        self.domain_mapping = {
            0: "health",
            1: "motor", 
            2: "travel"
        }
    
    def predict_domain(self, text: str) -> Tuple[str, float]:
        """
        Predict the domain of the input text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (domain, confidence_score)
        """
        try:
            # Get predictions from the model
            predictions = self.classifier(text)
            
            # Find the prediction with highest confidence
            best_prediction = max(predictions[0], key=lambda x: x['score'])
            
            # Map the label to domain name
            domain = self.domain_mapping.get(
                int(best_prediction['label'].split('_')[-1]), 
                "unknown"
            )
            
            confidence = best_prediction['score']
            
            return domain, confidence
            
        except Exception as e:
            # Fallback to default domain if classification fails
            print(f"Domain classification error: {e}")
            return "health", 0.0 