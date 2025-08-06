from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from services.clarification_service import ClarificationService
from services.slot_extractor import SlotExtractor

router = APIRouter()

# Pydantic models
class ClarificationUpdate(BaseModel):
    claim_id: str
    clarification_id: Optional[str] = None
    clarified_slots: Dict[str, any]  # slot_name -> clarified_value
    status: str = "confirmed"  # "confirmed" or "rejected"

class ClarificationResponse(BaseModel):
    claim_id: str
    status: str
    message: str
    updated_slots: Optional[Dict] = None

# Initialize services
clarification_service = ClarificationService()
slot_extractor = SlotExtractor()

@router.post("/claims/clarifications", response_model=ClarificationResponse)
async def receive_clarification(update: ClarificationUpdate):
    """
    Receive clarification updates from UI.
    
    1. Write back confirmed slot values
    2. Notify waiting intake process
    3. Return success
    """
    try:
        # 1. Resume processing with clarified slots
        resume_result = clarification_service.resume_processing(
            claim_id=update.claim_id,
            clarified_slots=update.clarified_slots
        )
        
        if resume_result["status"] == "processing_resumed":
            return ClarificationResponse(
                claim_id=update.claim_id,
                status="success",
                message="Clarification processed successfully",
                updated_slots=update.clarified_slots
            )
        else:
            return ClarificationResponse(
                claim_id=update.claim_id,
                status="error",
                message=f"Failed to resume processing: {resume_result.get('status')}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing clarification: {str(e)}"
        )

@router.get("/claims/clarifications/{claim_id}")
async def get_clarification_status(claim_id: str):
    """Get the status of a pending clarification."""
    try:
        clarification_status = clarification_service.check_clarification_status(claim_id)
        
        if clarification_status:
            return {
                "claim_id": claim_id,
                "status": "pending",
                "clarification_data": clarification_status
            }
        else:
            return {
                "claim_id": claim_id,
                "status": "not_found",
                "message": "No pending clarification found for this claim"
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking clarification status: {str(e)}"
        )

@router.put("/claims/clarifications/{claim_id}")
async def update_clarification(claim_id: str, update: ClarificationUpdate):
    """Update an existing clarification."""
    try:
        # This endpoint allows updating clarification data
        resume_result = clarification_service.resume_processing(
            claim_id=claim_id,
            clarified_slots=update.clarified_slots
        )
        
        return {
            "claim_id": claim_id,
            "status": "updated",
            "result": resume_result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating clarification: {str(e)}"
        ) 