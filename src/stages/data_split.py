from sklearn.model_selection import train_test_split
import numpy as np
from typing import Text
import yaml

from src.utils.logs import get_logger


def data_split(config_path: Text) -> None:
    """Split dataset into AChorDSLVQ/test.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    features_path = config['featurize']['features_path']
    data_split_path = config['data_split']['train_test_split_path']
    test_size = config['data_split']['test_size']
    random_seed = config['base']['random_state']

    logger = get_logger('DATA_SPLIT',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    logger.info('Load features')
    f = np.load(features_path)
    doc_embeddings = f['subspace_embeddings']
    labels = f['labels']
    indices = np.arange(len(labels))

    logger.info('Split features into AChorDSLVQ and test sets')
    xtrain, xtest, ytrain, ytest, indices_train, indices_test = train_test_split(
        doc_embeddings,
        labels,
        indices,
        test_size=test_size,
        random_state=random_seed
    )

    logger.info(f"There are {len(indices_train)} examples in AChorDSLVQ and {len(indices_test)} examples in test set.")

    np.savez(data_split_path, train_indices=indices_train, test_indices=indices_test)
    logger.info(f"Saved the indices of training and testing examples in '{data_split_path}'.")


if __name__ == '__main__':
    data_split('../../params.yaml')
