import logging
from google.cloud import storage
from app.config import Config, setup_logging
from google.cloud.storage import transfer_manager
from google.cloud.exceptions import GoogleCloudError
from google.cloud import bigquery
from datetime import datetime, timezone


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


        logger.info("✅ Adapter downloaded successfully from GCS.")

        return destination_directory

    except Exception as e:
        logger.exception(f"❌ Error downloading adapter from GCS: {e}. Will try to load from cache.")




def send_feedback_bq(
        user_claim: str, 
        predicted_category: int, 
        correct_category: int = None
        ):
    try:
        
        project_id = Config.GCP_PROJECT_ID
        if not project_id:
            raise ValueError("GCP_PROJECT_ID is not configured in Config.")

        client = bigquery.Client(project=project_id)        

        dataset_id = Config.BQ_DATASET_ID
        if not dataset_id:
            raise ValueError("BQ_DATASET_ID is not configured in Config.")

        table_id = Config.BQ_TABLE_ID
        if not table_id:
            raise ValueError("BQ_TABLE_ID is not configured in Config.")

        table_ref = client.dataset(dataset_id).table(table_id)

        table = client.get_table(table_ref)

        logger.info('BigQuery table : %s', table)
        
        row_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_claim": user_claim,
            "predicted_category": predicted_category,
            "correct_category": correct_category,
        }

        errors = client.insert_rows_json(table, [row_data])
        
        if errors:
            logger.error(f"BigQuery insertion failed: {errors}")
            raise Exception(f"BigQuery insertion failed: {errors}")
        
        logger.info(f"BigQuery insertion succeeded : {row_data}")
        
    except GoogleCloudError as e:
        logger.error(f"Google Cloud error while inserting feedback: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while inserting feedback: {e}")
        raise



if __name__ == '__main__':

    setup_logging()

    # load_model_gcs()
    send_feedback_bq(
        user_claim='Climate change is cool',
        predicted_category=3,
        correct_category=2
        )
