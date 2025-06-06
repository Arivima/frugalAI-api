fastapi-dev : 
	UV_ENV_FILE=".env" uv run uvicorn app.main:app --reload
#	uv run fastapi dev app/main.py

module-dev:
	UV_ENV_FILE=".env" uv run python -m app.gcp

