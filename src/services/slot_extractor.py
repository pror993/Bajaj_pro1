import yaml
import importlib
from typing import Dict, List, Tuple, Any
from pathlib import Path


class SlotExtractor:
    def __init__(self, profiles_path: str = "domain_profiles.yaml"):
        """
        Initialize slot extractor with domain profiles.
        
        Args:
            profiles_path: Path to the domain profiles YAML file
        """
        self.profiles = self._load_profiles(profiles_path)
        self.extractors = {}
        self._load_extractors()
    
    def _load_profiles(self, profiles_path: str) -> Dict:
        """Load domain profiles from YAML file."""
        try:
            with open(profiles_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading profiles: {e}")
            return {}
    
    def _load_extractors(self):
        """Dynamically import all extractor classes."""
        extractors_dir = Path(__file__).parent.parent / "extractors"
        
        for domain, config in self.profiles.get("domains", {}).items():
            for slot in config.get("slots", []):
                extractor_name = slot.get("extractor")
                if extractor_name:
                    try:
                        # Import the extractor module
                        module = importlib.import_module(f"src.extractors.{extractor_name.lower()}")
                        extractor_class = getattr(module, extractor_name)
                        self.extractors[extractor_name] = extractor_class()
                    except Exception as e:
                        print(f"Error loading extractor {extractor_name}: {e}")
    
    def extract_slots(self, text: str, domain: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Extract slots for a given domain and text.
        
        Args:
            text: Input text to extract slots from
            domain: Domain to extract slots for
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dict containing extracted slots and low confidence slots
        """
        if domain not in self.profiles.get("domains", {}):
            return {"slots": {}, "low_confidence_slots": []}
        
        extracted_slots = {}
        low_confidence_slots = []
        
        domain_config = self.profiles["domains"][domain]
        
        for slot in domain_config.get("slots", []):
            slot_name = slot["name"]
            extractor_name = slot["extractor"]
            
            if extractor_name in self.extractors:
                try:
                    value, confidence = self.extractors[extractor_name].extract(text)
                    
                    if confidence >= confidence_threshold:
                        extracted_slots[slot_name] = {
                            "value": value,
                            "confidence": confidence
                        }
                    else:
                        low_confidence_slots.append({
                            "slot_name": slot_name,
                            "detected_value": value,
                            "confidence": confidence
                        })
                        
                except Exception as e:
                    print(f"Error extracting slot {slot_name}: {e}")
        
        return {
            "slots": extracted_slots,
            "low_confidence_slots": low_confidence_slots
        } 