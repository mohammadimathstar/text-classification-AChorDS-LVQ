import numpy as np
from typing import Dict
import logging

from src.AChorDSLVQ.model import AChorDSLVQModel
from src.AChorDSLVQ.trainer import ModelTrainer


def train(doc_embeddings: np.ndarray,
          targets: np.ndarray,
          val_embeddings: np.ndarray,
          val_targets: np.ndarray,
          logger: logging.Logger,
          params: Dict):

    embedding_dim = doc_embeddings.shape[-2]
    subspace_dim = doc_embeddings.shape[-1]

    clf = AChorDSLVQModel(embedding_dim, subspace_dim, **params)
    trainer = ModelTrainer(clf, params)

    # Train the model
    trainer.train(xtrain=doc_embeddings,
                  ytrain=targets,
                  xval=val_embeddings,
                  yval=val_targets,
                  logger=logger,
                  **params)

    return clf

