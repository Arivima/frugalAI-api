import logging
from app.config import Config, setup_logging

logger = logging.getLogger('__name__')

def wandb_download():
    import wandb

    run = wandb.init()
    artifact = run.use_artifact(f'arivima-student/class_LLM_KD/{Config.ADAPTER_NAME}', type='model')
    artifact_dir = artifact.download()
    logger.info(f'artifact_dir: {artifact_dir}')
