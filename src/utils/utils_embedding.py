from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords
from typing import List, Tuple
from collections import Counter
import yaml
import tqdm
from src.utils.logs import get_logger
from src.utils.config_params import get_config
import os

import numpy as np
import pandas as pd



# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt_tab')


STOPWORDS = set(stopwords.words('english'))
MIN_TOKEN_LENGTH = 3


def preprocess_text(text: str) -> List[str]:
    """
    Tokenize and preprocess a text string by removing stopwords and non-alphabetic words,
    and applying length filtering.

    Parameters:
        text (str): The raw text to preprocess.

    Returns:
        List[str]: A list of cleaned and tokenized words.
    """
    tokens = nltk.word_tokenize(text.lower())
    return [token for token in tokens if token.isalpha() and token not in STOPWORDS and len(token) >= MIN_TOKEN_LENGTH]


def load_embedding_model(embedding_model_path) -> KeyedVectors:
    """
    Load a gensim word embedding model from disk.

    Parameters:
        embedding_model_path: the path to the gensim model (a '.d2v' file)

    Returns:
        KeyedVectors: The loaded gensim word embedding model.
    """

    return KeyedVectors.load(embedding_model_path)


def compute_document_embedding(model: KeyedVectors,
                               tokens: List[str],
                               subspace_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a document's embedding based on the word embeddings of its tokens.

    Parameters:
        model: the model loaded by gensim
        tokens (List[str]): A list of tokens (words).
        subspace_dim (int): The dimensionality of the embedding subspace

    Returns:
        Tuple[np.ndarray, np.ndarray]: The document embedding and a frequency vector.
    """
    token_counter = Counter(tokens)
    words_in_vocab = [word for word in token_counter.keys() if word in model.index_to_key]

    if len(words_in_vocab) >= subspace_dim:
        doc_embedding = np.array([model.get_vector(word) for word in words_in_vocab]).T
        word_frequencies = np.array([token_counter[word] for word in words_in_vocab])
    else:
        missing_words_count = subspace_dim - len(words_in_vocab)
        random_words = np.random.choice(words_in_vocab, size=missing_words_count, replace=True)
        words_in_vocab.extend([model.most_similar(word, topn=10)[np.random.randint(10)][0] for word in random_words])
        doc_embedding = np.array([model.get_vector(word) for word in words_in_vocab]).T
        doc_embedding += 0.00001 * np.random.randn(*doc_embedding.shape)
        word_frequencies = np.array([token_counter[word] for word in words_in_vocab] + [1] * missing_words_count)

    return doc_embedding, word_frequencies


def compute_document_embedding_full(model: KeyedVectors,
                               tokens: List[str],
                               subspace_dim: int) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Compute a document's embedding based on the word embeddings of its tokens.

    Parameters:
        model: the model loaded by gensim
        tokens (List[str]): A list of tokens (words).
        subspace_dim (int): The dimensionality of the embedding subspace

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The document embedding and a frequency vector.
    """
    token_counter = Counter(tokens)
    words_in_vocab = [word for word in token_counter.keys() if word in model.index_to_key]

    if len(words_in_vocab) >= subspace_dim:
        word_embeddings = np.array([model.get_vector(word) for word in words_in_vocab]).T
        word_frequencies = np.array([token_counter[word] for word in words_in_vocab])
    else:
        missing_words_count = subspace_dim - len(words_in_vocab)
        random_words = np.random.choice(words_in_vocab, size=missing_words_count, replace=True)
        words_in_vocab.extend([model.most_similar(word, topn=10)[np.random.randint(10)][0] for word in random_words])
        word_embeddings = np.array([model.get_vector(word) for word in words_in_vocab]).T
        word_embeddings += 0.00001 * np.random.randn(*word_embeddings.shape)
        word_frequencies = np.array([token_counter[word] for word in words_in_vocab] + [1] * missing_words_count)

    return word_embeddings, word_frequencies, words_in_vocab


def get_document_subspace(doc_embedding: np.ndarray,
                          word_frequencies: np.ndarray,
                          subspace_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the document subspace using Singular Value Decomposition (SVD).

    Parameters:
        doc_embedding (np.ndarray): The document embedding matrix.
        word_frequencies (np.ndarray): Frequency of words in the document.
        subspace_dim (int): The dimensionality of the embedding subspace

    Returns:
        np.ndarray: The subspace matrix.
    """

    U, S, _ = np.linalg.svd(doc_embedding * np.sqrt(word_frequencies), full_matrices=False)

    return U[:, :subspace_dim], S


def get_document_subspace_full(doc_embedding: np.ndarray,
                               word_frequencies: np.ndarray,
                               subspace_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the document subspace using Singular Value Decomposition (SVD).

    Parameters:
        doc_embedding (np.ndarray): The document embedding matrix.
        word_frequencies (np.ndarray): Frequency of words in the document.
        subspace_dim (int): The dimensionality of the embedding subspace

    Returns:
        np.ndarray: The subspace matrix.
    """
    U, S, Vh = np.linalg.svd(doc_embedding * np.sqrt(word_frequencies), full_matrices=False)

    return U[:, :subspace_dim], S, Vh


def generate_document_embeddings(model: KeyedVectors,
                                 corpus_tokens: List[List[str]],
                                 embedding_dim: int,
                                 subspace_dim: int,
                                 return_signular_values=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate embeddings for a list of documents.

    Parameters:
        model: the gensim model
        corpus_tokens (List[List[str]]): A list of tokenized documents.
        embedding_dim: the dimensionality of vector embeddings (e.g. 300)
        subspace_dim (int): The dimensionality of the embedding subspace
        return_signular_values (np.ndarray): if singular values (for each document) should be returned

    Returns:
        np.ndarray: A matrix containing the document embeddings.
    """

    document_embeddings = np.zeros((len(corpus_tokens), embedding_dim, subspace_dim))
    if return_signular_values:
        document_singular_values = np.zeros((len(corpus_tokens), subspace_dim))
    else:
        document_singular_values = None

    for i, tokens in tqdm.tqdm(enumerate(corpus_tokens), total=len(corpus_tokens), desc="Processing documents"):
        doc_embedding, word_frequencies = compute_document_embedding(model, tokens, subspace_dim)
        document_embeddings[i], S = get_document_subspace(doc_embedding, word_frequencies, subspace_dim)
        if return_signular_values:
            document_singular_values[i] = S

    return document_embeddings, document_singular_values


def preprocess_corpus(model,
                      corpus_text: List[str],
                      embedding_dim,
                      subspace_dim,
                      logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a corpus of text and generate embeddings for each document.

    Parameters:
        model: the gensim model (a glove model)
        corpus_text (List[str]): A list of raw text documents.
        embedding_dim: The dimensionality of the vector embedding (e.g. 300)
        subspace_dim: The dimensionality of the embedding subspace

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - The generated document embeddings.
            - The singular values of the document embeddings (if computed).
    """

    logger.info(f"Starting preprocessing of {len(corpus_text)} documents.")

    # Preprocess the corpus by tokenizing the text
    corpus_tokens = [preprocess_text(doc) for doc in corpus_text]
    logger.info("All documents have been tokenized.")

    # Generate document embeddings and the singular values (if any)
    document_embeddings, document_singular_values = generate_document_embeddings(model,
                                                                                 corpus_tokens,
                                                                                 embedding_dim,
                                                                                 subspace_dim)
    logger.info(f"Document embeddings (subspace embeddings) with dimensionality {subspace_dim} have been computed.")

    return document_embeddings, document_singular_values
