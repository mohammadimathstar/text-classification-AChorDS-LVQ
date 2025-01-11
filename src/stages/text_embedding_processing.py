
import os
import argparse
import numpy as np
import pandas as pd
import yaml

from src.utils.utils_embedding import preprocess_corpus, load_embedding_model
from src.utils.logs import get_logger


def preprocess_data(config_path):

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('EMBEDDING',
                        log_level=config['base']['log_level'],
                        log_file=config['base']['log_file'])

    logger.info(f"Load word embedding model: '{config['featurize']['embedding_model']}'.")
    model = load_embedding_model(config['featurize']['embedding_model'])

    logger.info(f"Load data set from '{config['load_data']['dataset_csv']}'.")
    df = pd.read_csv(config['load_data']['dataset_csv'])
    corpus = df[config['featurize']['text_column']].tolist()
    labels = df[config['featurize']['target_column']].tolist()

    # Compute embedding subspaces for documents
    doc_embeddings, _ = preprocess_corpus(model,
                                          corpus,
                                          config['featurize']['embedding_dim'],
                                          config['featurize']['subspace_dim'],
                                          logger)

    # Save the embedding subspaces
    data_dir = "/".join(config['featurize']['features_path'].split("/")[:-1])
    os.makedirs(data_dir, exist_ok=True)
    np.savez(config['featurize']['features_path'], subspace_embeddings=doc_embeddings, labels=labels)
    logger.info(f"Documents embeddings have been saved in '{config['featurize']['features_path']}'.")


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    preprocess_data(config_path=args.config)