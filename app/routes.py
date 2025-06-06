import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

router = APIRouter()


class ClassifyRequest(BaseModel):
    user_claim: str = Field(..., strip_whitespace=True, min_length=1)

class ClassifyResponse(BaseModel):
    model_name: str
    user_claim: str
    category: str
    explanation : str


@router.get("/")
async def root():
    return {"status": "ok"}


@router.post("/classify", response_model=ClassifyResponse)
async def classify(request: Request, body: ClassifyRequest):

    logger.info("New request")

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