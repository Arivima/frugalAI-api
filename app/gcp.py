import re
import json
import logging
from google.cloud import storage
from app.config import Config, setup_logging
from google.cloud.storage import transfer_manager

logger = logging.getLogger('__name__')
DESTINATION_DIRECTORY = 'models'

def load_model_gcs():
    try:
        logger.info("load_model_gcs")

        project_id = Config.GCP_PROJECT_ID
        logger.info('project_id %s', project_id)
        if not project_id:
            raise ValueError("GCP_PROJECT_ID is not configured in Config.")
        client = storage.Client(project=project_id)
        logger.info('connected to client %s', client.project)

        bucket_name = Config.GCS_BUCKET_NAME
        logger.info('bucket_name %s', bucket_name)
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME is not configured in Config.")
        bucket = client.bucket(bucket_name)
        logger.info('connected to bucket %s', bucket.name)


        blobs = list(bucket.list_blobs(
            prefix='model_KD_student_CE:v11/',
            delimiter="/"
            ))
        logger.info(f'Number of blobs found: {len(blobs)}')
        if not blobs:
            logger.error(f"No model files found in GCS")
            return None

        blob_names = [blob.name for blob in blobs]

        results = transfer_manager.download_many_to_path(
            bucket, blob_names, destination_directory=DESTINATION_DIRECTORY
        )

        for name, result in zip(blob_names, results):
            if isinstance(result, Exception):
                logger.error("Failed to download %s due to exception: %s", name, result)
            else:
                logger.info("Downloaded %s.", DESTINATION_DIRECTORY + '/' + name)


        logger.info("✅ Model weights and intercept loaded successfully from GCS.")
        return results

    except Exception as e:
        logger.exception(f"❌ Error loading model from GCS bucket: {e}")
        raise


if __name__ == '__main__':
    setup_logging()

    load_model_gcs()

    # import wandb

    # run = wandb.init()
    # artifact = run.use_artifact('arivima-student/class_LLM_KD/model_KD_student_CE:v11', type='model')
    # artifact_dir = artifact.download()
    # logger.info(f'artifact_dir: {artifact_dir}')
