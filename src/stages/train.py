
import numpy as np
import yaml

from src.AChorDSLVQ.training import train
from src.utils.logs import get_logger
from src.utils.utils import save_model
import argparse


def train_model(config_path: str) -> None:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('TRAIN',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    logger.info('Load AChorDSLVQ dataset')

    train_indices = np.load(config['data_split']['train_test_split_path'])['train_indices']
    test_indices = np.load(config['data_split']['train_test_split_path'])['test_indices']
    data = np.load(config['featurize']['features_path'])

    train_embeddings = data['subspace_embeddings'][train_indices]
    train_labels = data['labels'][train_indices]

    val_embeddings = data['subspace_embeddings'][test_indices]
    val_labels = data['labels'][test_indices]

    logger.info('Train model')
    clf = train(
        doc_embeddings=train_embeddings,
        targets=train_labels,
        val_embeddings=val_embeddings,
        val_targets=val_labels,
        logger=logger,
        params=config['train']['hyperparams'],
    )
    # logger.info(f'Best score: {model.best_score_}')

    logger.info('Save model')
    model_path = config['train']['model_path']
    save_model(clf, model_path)
    print(f"Model saved to '{model_path}'.")


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
