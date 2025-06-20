import logging
from pydantic import BaseModel, Field
from fastapi import (
    APIRouter, 
    Request, 
    Response,
    HTTPException, 
    status
    )
from app.gcp import send_feedback_bq

logger = logging.getLogger(__name__)

router = APIRouter()


class ClassifyRequest(BaseModel):
    user_claim: str = Field(..., strip_whitespace=True, min_length=1)

class ClassifyResponse(BaseModel):
    model_name: str
    user_claim: str
    category: str
    explanation : str

class FeedbackRequest(BaseModel):
    user_claim: str = Field(..., strip_whitespace=True, min_length=1)
    predicted_category: int = Field(..., ge=0, le=7)
    assistant_explanation: str = Field(..., strip_whitespace=True, min_length=1)
    correct_category: int = Field(..., ge=0, le=7)


@router.get("/")
async def root():
    return {"status": "ok"}


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: Request, body: ClassifyRequest):

    logger.info(f"New classification request : {body}")

    llm = getattr(request.app.state, "model", None)
    if llm is None:
        logger.error(f"Model not available")
        raise HTTPException(status_code=500, detail="Model not available")
    logger.info(f"Model available")

    try:
        category, explanation = llm.generate(quote=body.user_claim)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")

    response_data = {
        "model_name":   llm.model_name,
        "user_claim":   body.user_claim,
        "category":     category,
        "explanation":  explanation
    }
    logger.info(f"response: {response_data}")
    return ClassifyResponse(**response_data)


@router.post("/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def submit_feedback(request: Request, body: FeedbackRequest):

    logger.info(f"New feedback request : {body}")

    send_feedback_bq(
        user_claim=body.user_claim,
        predicted_category=int(body.predicted_category),
        assistant_explanation=body.assistant_explanation,
        correct_category=int(body.correct_category)
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)

