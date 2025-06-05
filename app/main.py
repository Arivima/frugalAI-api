# TODO V1
# - load model from gcp
# - tests
# - input validation

import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import router
from app.config import setup_logging
from app.gcp import load_model_gcs


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Starting API')
    models = {}
    models["distilled"] = load_model_gcs()
    yield

    models.clear()
    print('Shutting down API')


def main():

    setup_logging()
    logger = logging.getLogger(__name__)

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)


if __name__ == "__main__":
    main()