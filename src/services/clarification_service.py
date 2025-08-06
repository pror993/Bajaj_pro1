import requests
import json
from typing import Dict, List, Optional
from datetime import datetime


class ClarificationService:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize clarification service.
        
        Args:
            api_base_url: Base URL for the clarifications API
        """
        self.api_base_url = api_base_url
        self.clarification_endpoint = f"{api_base_url}/api/claims/clarifications"
        self.pending_clarifications = {}
    
    def process_low_confidence_slots(
        self, 
        claim_id: str, 
        low_confidence_slots: List[Dict], 
        threshold: float = 0.7
    ) -> Dict:
        """
        Process low confidence slots and request clarifications.
        
        Args:
            claim_id: Unique identifier for the claim
            low_confidence_slots: List of slots with low confidence
            threshold: Confidence threshold for requiring clarification
            
        Returns:
            Dict with clarification status and pending slots
        """
        if not low_confidence_slots:
            return {"status": "no_clarification_needed", "pending_slots": []}
        
        # Filter slots below threshold
        slots_needing_clarification = [
            slot for slot in low_confidence_slots 
            if slot.get("confidence", 0) < threshold
        ]
        
        if not slots_needing_clarification:
            return {"status": "no_clarification_needed", "pending_slots": []}
        
        # Prepare clarification payload
        clarification_payload = {
            "claim_id": claim_id,
            "timestamp": datetime.now().isoformat(),
            "slots_requiring_clarification": slots_needing_clarification,
            "status": "pending"
        }
        
        try:
            # Post to clarifications API
            response = requests.post(
                self.clarification_endpoint,
                json=clarification_payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                # Store pending clarification
                self.pending_clarifications[claim_id] = {
                    "payload": clarification_payload,
                    "status": "pending",
                    "created_at": datetime.now()
                }
                
                return {
                    "status": "clarification_requested",
                    "pending_slots": [slot["slot_name"] for slot in slots_needing_clarification],
                    "clarification_id": response.json().get("clarification_id")
                }
            else:
                return {
                    "status": "clarification_failed",
                    "error": f"API returned status {response.status_code}",
                    "pending_slots": []
                }
                
        except Exception as e:
            return {
                "status": "clarification_failed",
                "error": str(e),
                "pending_slots": []
            }
    
    def check_clarification_status(self, claim_id: str) -> Optional[Dict]:
        """
        Check the status of a pending clarification.
        
        Args:
            claim_id: Claim ID to check status for
            
        Returns:
            Clarification status or None if not found
        """
        if claim_id not in self.pending_clarifications:
            return None
        
        try:
            response = requests.get(
                f"{self.clarification_endpoint}/{claim_id}"
            )
            
            if response.status_code == 200:
                status_data = response.json()
                self.pending_clarifications[claim_id]["status"] = status_data.get("status")
                return status_data
            else:
                return None
                
        except Exception as e:
            print(f"Error checking clarification status: {e}")
            return None
    
    def resume_processing(self, claim_id: str, clarified_slots: Dict) -> Dict:
        """
        Resume processing after clarification is received.
        
        Args:
            claim_id: Claim ID to resume processing for
            clarified_slots: Dict of slot_name -> clarified_value
            
        Returns:
            Updated processing status
        """
        if claim_id in self.pending_clarifications:
            # Update the pending clarification with clarified values
            self.pending_clarifications[claim_id]["clarified_slots"] = clarified_slots
            self.pending_clarifications[claim_id]["status"] = "resolved"
            
            return {
                "status": "processing_resumed",
                "claim_id": claim_id,
                "clarified_slots": clarified_slots
            }
        else:
            return {
                "status": "claim_not_found",
                "claim_id": claim_id
            } 