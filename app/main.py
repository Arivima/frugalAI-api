import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.config import setup_logging
from app.gcp import load_model_gcs
from app.model import LLMWrapper

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Starting API')
    try:
        load_model_gcs()
        app.state.model = LLMWrapper()

    except Exception as e:
        logger.exception(f"{e}")

    yield

    if app.state.model is not None:
        app.state.model.clear()
    logger.info("Shutting down API")


app = FastAPI(lifespan=lifespan)
setup_logging()
app.include_router(router)
