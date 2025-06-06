import logging
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

router = APIRouter()


class ClassifyRequest(BaseModel):
    claim: str = Field(..., min_length=1)

class ClassifyResponse(BaseModel):
    classification: str


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
        output = llm.generate(prompt=body.claim)
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error during generation: {e}")

    return ClassifyResponse(classification=output)
