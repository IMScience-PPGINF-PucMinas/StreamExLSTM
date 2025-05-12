import logging

from configs.constants import SUMME_DATASET_PATH, SUMME_LOG_DIR, SUMME_SAVE_DIR, SUMME_SCORE_DIR, SUMME_SPLIT_FILE_PATH, TVSUM_DATASET_PATH, TVSUM_LOG_DIR, TVSUM_SAVE_DIR, TVSUM_SCORE_DIR, TVSUM_SPLIT_FILE_PATH

def setup_logging(level=logging.INFO):
    """Set up logging with a consistent format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def get_paths(dataset_name):
    if dataset_name == 'SumMe':
        return {
            'dataset': SUMME_DATASET_PATH,
            'split': SUMME_SPLIT_FILE_PATH,
            'save_dir': SUMME_SAVE_DIR,
            'log_dir': SUMME_LOG_DIR,
            'score_dir': SUMME_SCORE_DIR
        }
    elif dataset_name == 'TVSum':
        return {
            'dataset': TVSUM_DATASET_PATH,
            'split': TVSUM_SPLIT_FILE_PATH,
            'save_dir': TVSUM_SAVE_DIR,
            'log_dir': TVSUM_LOG_DIR,
            'score_dir': TVSUM_SCORE_DIR
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
