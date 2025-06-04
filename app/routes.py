from fastapi import APIRouter


router = APIRouter(
    tags=["Classification"]
)


@router.get("/")
async def root():
    return {"status": "ok"}

@router.post("/classify")
async def classify(request):
    return 
