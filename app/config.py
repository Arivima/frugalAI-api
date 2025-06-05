# app/config.py
import os
import logging

class Config:
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

