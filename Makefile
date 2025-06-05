fastapi-dev : 
	uv run fastapi dev app/main.py
# uv run uvicorn app.main:app --reload

module-dev:
	UV_ENV_FILE=".env" uv run python -m app.gcp

