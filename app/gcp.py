import logging
from google.cloud import storage
from app.config import Config, setup_logging
from google.cloud.storage import transfer_manager

logger = logging.getLogger(__name__)


def load_model_gcs():
    try:
        logger.info("load_model_gcs")

        project_id = Config.GCP_PROJECT_ID
        if not project_id:
            raise ValueError("GCP_PROJECT_ID is not configured in Config.")
        client = storage.Client(project=project_id)
        logger.info('connected to client : %s', client.project)

        bucket_name = Config.GCS_BUCKET_NAME
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME is not configured in Config.")
        bucket = client.bucket(bucket_name)
        logger.info('connected to bucket : %s', bucket.name)

        prefix = Config.ADAPTER_NAME + '/'
        logger.info(f'Looking with prefix : {prefix}')
        blobs = list(bucket.list_blobs(
            prefix=prefix,
            delimiter="/"
            ))
        logger.info(f'Number of blobs found: {len(blobs)}')
        if not blobs:
            raise ValueError("No files found")

        blob_names = [blob.name for blob in blobs]

        destination_directory = Config.DESTINATION_DIRECTORY
        results = transfer_manager.download_many_to_path(
            bucket, blob_names, destination_directory=destination_directory
        )

        for name, result in zip(blob_names, results):
            if isinstance(result, Exception):
                logger.error("Failed to download %s due to exception: %s", name, result)
            else:
                logger.info("Downloaded %s.", destination_directory + '/' + name)


        logger.info("✅ Adapter loaded successfully from GCS.")

        return destination_directory

    except Exception as e:
        logger.exception(f"❌ Error loading model from GCS bucket: {e}. Will try to load from cache.")



if __name__ == '__main__':
    setup_logging()

    load_model_gcs()

