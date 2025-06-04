from fastapi import FastAPI
from app.routes import router
from contextlib import asynccontextmanager


def fake_model(x: float):
    return x * 42

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('hello')
    models["distilled"] = fake_model(32)
    yield

    models.clear()
    print('bye')

app = FastAPI(lifespan=lifespan)

app.include_router(router)