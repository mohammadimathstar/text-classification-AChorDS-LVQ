import yaml
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from typing import Tuple, Dict

from src.utils.utils import load_model, find_closest_prototypes_without_label
from src.utils.logs import get_logger
from src.utils.utils_embedding import (load_embedding_model,
                                       compute_document_embedding_full,
                                       preprocess_text, get_document_subspace_full)
from src.AChorDSLVQ.model import AChorDSLVQModel
from src.report.visualization_explainability import plot_top_words


def compute_subspace_contribution(vh, s, word_frequencies, subspace_dim):
    """
    Parameters:
        Vh: (D x n)-matrix - (n: number of distinct words in the doc, D: embedding dimensionality e.g.300).
        S: (n x n)- matrix - a vector containing  singular values.
        word_frequencies (1 x n)-matrix - The number of times a word in the text appears

    Return:
         RS^(-1) : it is a (n x d) matrix
    It also shows the coordinate of each word on the reconstructed subspace
    """

    if word_frequencies.ndim == 1 :
        word_frequencies = np.expand_dims(word_frequencies, axis=1)
    RS_1 = word_frequencies * vh[:subspace_dim,:].T @ np.diag(1 / s[:subspace_dim])

    return RS_1


def compute_documbent_embedding(txt: str,
                                embedding_model: KeyedVectors,
                                subspace_dim: int) -> Dict:
    """
    Compute document embedding using a word embedding method e.g. GloVe

    Parameters:
        txt (str): the text of a document
        embedding_model: the word embedding model e.g. GloVe or Word2Vec
        subspace_dim (int): the dimensionality of subspace embedding

    Returns:
        a dictionary containing document embedding and the impact of each word on it.
    """

    # Preprocess data
    tokens = preprocess_text(txt)
    word_embeddings, word_frequencies, words = compute_document_embedding_full(embedding_model,
                                                                             tokens,
                                                                             subspace_dim)
    # Compute Document Embedding
    doc_embedding, s, vh = get_document_subspace_full(word_embeddings, word_frequencies, subspace_dim)
    word_impact_on_doc_embedding = compute_subspace_contribution(vh, s, word_frequencies, subspace_dim)

    return {
        'doc_embedding': doc_embedding,
        'word_impact_on_doc_embedding': word_impact_on_doc_embedding,
        'words': words,
        'word_embeddings': word_embeddings,
    }


def prediction_with_winners(doc_embedding: np.ndarray,
                            achords_model: AChorDSLVQModel) -> Tuple[Dict, int, int]:
    """
    Find the closest two closest prototypes to a document

    Parameters:
        doc_embedding: the embedding subspace of the document
        achords_model: the classifier model

    Returns:
        a dictionary containing the closest two prototypes and model's prediction
    """

    # Prediction and Word Weights
    first_winner, second_winner = find_closest_prototypes_without_label(doc_embedding,
                                                                        achords_model.prototype_features,
                                                                        achords_model.prototype_labels,
                                                                        achords_model.lambda_value)
    return {
        'closest_prototype': first_winner,
        'second_closest_prototype': second_winner,
    }, achords_model.prototype_labels[first_winner['index']], achords_model.prototype_labels[second_winner['index']]


def get_word_importances(txt: str,
                         embedding_model: KeyedVectors,
                         achords_model: AChorDSLVQModel):

    subspace_dim = achords_model.prototype_features.shape[-1]

    # Compute document subspace embedding
    doc_repr = compute_documbent_embedding(txt, embedding_model, subspace_dim)

    # Do prediction and find two closest prototypes
    winners, pred, second_pred = prediction_with_winners(doc_repr['doc_embedding'], achords_model)

    word_importances_winners = {}
    for winner_type, dic in winners.items():
        X, M = doc_repr['word_embeddings'], doc_repr['word_impact_on_doc_embedding'] @ dic['Q']
        W = X.T @ achords_model.prototype_features[dic['index']] @ dic['Qw']
        txt_words_impact = achords_model.lambda_value * M * W
        word_importances_winners[winner_type] = txt_words_impact.sum(axis=1)

    return word_importances_winners, doc_repr['words'], pred, second_pred


def get_top_words(words,
                  word_importance_for_winners,
                  num_of_top_words,
                  logger):

    positive_scores = word_importance_for_winners['closest_prototype']
    negative_scores = word_importance_for_winners['second_closest_prototype']
    diff = positive_scores - negative_scores
    sort_idx_pos = np.argsort(positive_scores)
    sort_idx_neg = np.argsort(negative_scores)
    sort_idx_diff = np.argsort(diff)
    top_words = [(i, words[i]) for i in sort_idx_pos[-1:-num_of_top_words-1:-1]]
    top_words.extend([(i, words[i]) for i in sort_idx_neg[-1:-num_of_top_words-1:-1]])
    top_words.extend([(i, words[i]) for i in sort_idx_diff[-1:-num_of_top_words-1:-1]])
    top_words_set = set(top_words)
    logger.info(f"There are {len(top_words_set)} distinct top words among ({num_of_top_words} top words with respect to two closest prototypes).")
    dic = dict()
    for i, w in top_words_set:
        dic[w] = {
            'closest_prototype_score': positive_scores[i],
            'second_closest_prototype_score': negative_scores[i],
            'decision': diff[i],
        }
    logger.info(f'{num_of_top_words} top impactful words (on decision): ' + str({words[i]: diff[i] for i in sort_idx_diff[-1:-num_of_top_words:-1]}))

    return pd.DataFrame.from_dict(dic, orient='index')



def explain_decision(txt: str, config_path: str, address_prefix: str = "") -> None:

    with open(address_prefix + config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    label2class = {i: name for i, name in enumerate(config['evaluate']['target_names'])}

    logger = get_logger("EXPLAIN",
                        log_level=config['base']['log_level'],
                        log_file=address_prefix + config['base']['log_file'])

    # Load embedding model
    logger.info("Load embedding model")
    embedding_model = load_embedding_model(
        embedding_model_path=address_prefix + config['featurize']['embedding_model'])

    # Load AChorDS-LVQ model
    achords_model = load_model(filepath=address_prefix + config['train']['model_path'])

    # Compute words' importances
    word_importances_for_winners, words, pred, second_pred = get_word_importances(txt,
                                                                                  embedding_model,
                                                                                  achords_model)
    if label2class:
        logger.info(f"Predicted class: '{label2class[pred]}', \t Predicted label: {pred}.")
    else:
        logger.info(f"Predicted label: {pred}")

    top_words_df = get_top_words(words,
                                 word_importances_for_winners,
                                 config['explainability']['ntops'],
                                 logger)

    top_words_df.rename(columns={'closest_prototype_score' : label2class[pred] if label2class else pred,
                        'second_closest_prototype_score': label2class[second_pred] if label2class else second_pred},
                        inplace=True)

    top_words_df.index.names = ['Words']
    top_words_df.to_csv(address_prefix + config['explainability']['table_path'])
    logger.info(f"Top {config['explainability']['ntops']} words are save in '{address_prefix + config['explainability']['table_path']}'")

    plot_top_words(top_words_df, address_prefix + config['explainability']['fig_path'])
    logger.info(f"The visualization of {config['explainability']['ntops']} words are save in '{address_prefix + config['explainability']['fig_path']}'")



if __name__ == '__main__':
    df = pd.read_csv('../../data/raw/dataset.csv')
    corpus = df['text'].tolist()

    txt = corpus[1]

    explain_decision(txt,
                     config_path='params.yaml',
                     address_prefix="../../")
