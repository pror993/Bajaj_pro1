from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from services.domain_classifier import DomainClassifier
from services.slot_extractor import SlotExtractor
from services.clarification_service import ClarificationService

router = APIRouter()

# Pydantic models
class IntakeRequest(BaseModel):
    claim_text: str
    claim_id: Optional[str] = None
    confidence_threshold: float = 0.7

class IntakeResponse(BaseModel):
    claim_id: str
    domain: str
    domain_confidence: float
    extracted_slots: Dict
    low_confidence_slots: List[Dict]
    status: str  # "completed" or "pending_clarification"
    clarification_id: Optional[str] = None

# Initialize services
domain_classifier = DomainClassifier()
slot_extractor = SlotExtractor()
clarification_service = ClarificationService()

@router.post("/claims/intake", response_model=IntakeResponse)
async def intake_claim(request: IntakeRequest):
    """
    Process a new claim intake.
    
    1. Predict domain
    2. Extract slots
    3. If any low-confidence → call clarification_service
    4. Return initial structured claim or "pending clarification"
    """
    try:
        # Generate claim ID if not provided
        claim_id = request.claim_id or str(uuid.uuid4())
        
        # 1. Predict domain
        domain, domain_confidence = domain_classifier.predict_domain(request.claim_text)
        
        # 2. Extract slots
        extraction_result = slot_extractor.extract_slots(
            text=request.claim_text,
            domain=domain,
            confidence_threshold=request.confidence_threshold
        )
        
        extracted_slots = extraction_result["slots"]
        low_confidence_slots = extraction_result["low_confidence_slots"]
        
        # 3. Handle low confidence slots
        clarification_result = None
        status = "completed"
        clarification_id = None
        
        if low_confidence_slots:
            clarification_result = clarification_service.process_low_confidence_slots(
                claim_id=claim_id,
                low_confidence_slots=low_confidence_slots,
                threshold=request.confidence_threshold
            )
            
            if clarification_result["status"] == "clarification_requested":
                status = "pending_clarification"
                clarification_id = clarification_result.get("clarification_id")
        
        # 4. Return response
        return IntakeResponse(
            claim_id=claim_id,
            domain=domain,
            domain_confidence=domain_confidence,
            extracted_slots=extracted_slots,
            low_confidence_slots=low_confidence_slots,
            status=status,
            clarification_id=clarification_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing claim intake: {str(e)}"
        )

@router.get("/claims/{claim_id}/status")
async def get_claim_status(claim_id: str):
    """Get the status of a claim processing."""
    try:
        # Check clarification status if pending
        clarification_status = clarification_service.check_clarification_status(claim_id)
        
        if clarification_status:
            return {
                "claim_id": claim_id,
                "status": clarification_status.get("status", "unknown"),
                "clarification_data": clarification_status
            }
        else:
            return {
                "claim_id": claim_id,
                "status": "completed",
                "message": "No pending clarifications found"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking claim status: {str(e)}"
        ) 