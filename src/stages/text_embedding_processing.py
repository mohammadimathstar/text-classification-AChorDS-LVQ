
import gensim.downloader as gensim_api
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


def load_embedding_model() -> KeyedVectors:
    """
    Load a gensim word embedding model from disk.

    Returns:
        KeyedVectors: The loaded gensim word embedding model.
    """

    return KeyedVectors.load(EMBEDDING_MODEL)


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


def compute_document_embedding(tokens: List[str]) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Compute a document's embedding based on the word embeddings of its tokens.

    Parameters:
        tokens (List[str]): A list of tokens (words).

    Returns:
        Tuple[np.ndarray, np.ndarray]: The document embedding and a frequency vector.
    """
    token_counter = Counter(tokens)
    words_in_vocab = [word for word in token_counter.keys() if word in model.index_to_key]

    if len(words_in_vocab) >= SUBSPACE_DIM:
        doc_embedding = np.array([model.get_vector(word) for word in words_in_vocab]).T
        word_frequencies = np.array([token_counter[word] for word in words_in_vocab])
    else:
        missing_words_count = SUBSPACE_DIM - len(words_in_vocab)
        random_words = np.random.choice(words_in_vocab, size=missing_words_count, replace=True)
        words_in_vocab.extend([model.most_similar(word, topn=10)[np.random.randint(10)][0] for word in random_words])
        doc_embedding = np.array([model.get_vector(word) for word in words_in_vocab]).T
        doc_embedding += 0.00001 * np.random.randn(*doc_embedding.shape)
        word_frequencies = np.array([token_counter[word] for word in words_in_vocab] + [1] * missing_words_count)

    return doc_embedding, word_frequencies


def get_document_subspace(doc_embedding: np.ndarray, word_frequencies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the document subspace using Singular Value Decomposition (SVD).

    Parameters:
        doc_embedding (np.ndarray): The document embedding matrix.
        word_frequencies (np.ndarray): Frequency of words in the document.

    Returns:
        np.ndarray: The subspace matrix.
    """
    U, S, _ = np.linalg.svd(doc_embedding * np.sqrt(word_frequencies), full_matrices=False)

    return U[:, :SUBSPACE_DIM], S


def generate_document_embeddings(corpus_tokens: List[List[str]],
                                 return_signular_values=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate embeddings for a list of documents.

    Parameters:
        corpus_tokens (List[List[str]]): A list of tokenized documents.
        return_signular_values (np.ndarray): if singular values (for each document) should be returned
    Returns:
        np.ndarray: A matrix containing the document embeddings.
    """

    document_embeddings = np.zeros((len(corpus_tokens), EMBEDDING_DIM, SUBSPACE_DIM))
    if return_signular_values:
        document_singular_values = np.zeros((len(corpus_tokens), SUBSPACE_DIM))
    else:
        document_singular_values = None

    for i, tokens in tqdm.tqdm(enumerate(corpus_tokens), total=len(corpus_tokens), desc="Processing documents"):
        doc_embedding, word_frequencies = compute_document_embedding(tokens)
        document_embeddings[i], S = get_document_subspace(doc_embedding, word_frequencies)
        if return_signular_values:
            document_singular_values[i] = S


    return document_embeddings, document_singular_values


def preprocess_corpus(corpus_text: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a corpus of text and generate embeddings for each document.

    Parameters:
        corpus_text (List[str]): A list of raw text documents.

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
    document_embeddings, document_singular_values = generate_document_embeddings(corpus_tokens)
    logger.info(f"Document embeddings (subspace embeddings) with dimensionality {SUBSPACE_DIM} have been computed.")

    return document_embeddings, document_singular_values


config = get_config(config_path='./params.yaml')
logger = get_logger('EMBEDDING', log_level=config['base']['log_level'])

INPUT_DATA_PATH = config['load_data']['dataset_csv']
PROCESSED_DATA_PATH = config['featurize']['features_path']
TEXT_COL = config['featurize']['text_column']
LABEL_COL = config['featurize']['target_column']
SUBSPACE_DIM = config['featurize']['subspace_dim']
EMBEDDING_DIM = config['featurize']['embedding_dim']
MIN_TOKEN_LENGTH = config['featurize']['min_token_length']
EMBEDDING_MODEL = config['featurize']['embedding_model']

STOPWORDS = set(stopwords.words('english'))

model = load_embedding_model()
df = pd.read_csv(INPUT_DATA_PATH)
corpus = df[TEXT_COL].tolist()
labels = df[LABEL_COL].tolist()

# Compute embedding subspaces for documents
doc_embeddings, _ = preprocess_corpus(corpus)

# Save the embedding subspaces
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
np.savez(PROCESSED_DATA_PATH, subspace_embeddings=doc_embeddings, labels=labels)
logger.info(f"Documents embeddings have been saved in '{PROCESSED_DATA_PATH}'.")
