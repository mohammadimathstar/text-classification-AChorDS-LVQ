import os

import numpy as np
from scipy.linalg import orth
from typing import List, Union
import pickle
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


def orthogonalize_data(data_3d: np.ndarray):
    """
    Applies orthogonalization to the data in the given 3D array.

    Args:
    - data_3d (ndarray): A 3D array where each element is a 2D array representing data to be orthogonalized.

    Returns:
    - ndarray: The 3D array where each data element is orthogonalized.
    """
    for i, data in enumerate(data_3d):
        data_3d[i] = orth(data)  # Using numpy's orthogonalization
    return data_3d


# --------------------------- Prototype Initialization ---------------------------

def initialize_random_prototypes(embedding_dim: int,
                                 subspace_dim: int,
                                 labels: Union[int, List, np.ndarray],
                                 num_prototypes: Union[int, List[int]] = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize random prototypes using Gaussian distribution and orthogonalize them.

    Args:
        embedding_dim (int): Dimensionality of the word-embedding model.
        subspace_dim (int): Dimensionality of the subspace.
        labels (List or np.ndarray): List of class labels.
        num_prototypes (int or List[int], optional): Number of prototypes per class. Defaults to 1.

    Returns:
        np.ndarray: Initialized prototypes.
        np.ndarray: Corresponding class labels for prototypes.
    """
    if isinstance(labels, int):
        labels = np.arange(labels)

    unique_classes = np.unique(labels)
    total_prototypes = np.sum(num_prototypes) if isinstance(num_prototypes, list) else len(unique_classes) * num_prototypes

    prototypes = np.random.normal(0, 1, (total_prototypes, embedding_dim, subspace_dim))

    # orthogonalize prototypes
    prototypes = orthogonalize_data(prototypes)

    prototype_labels = np.repeat(unique_classes, num_prototypes)

    return prototypes, prototype_labels


def initialize_prototypes_from_data(data: np.ndarray,
                                    labels: Union[List, np.ndarray],
                                    num_prototypes: Union[int, List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize prototypes directly from data points.

    Args:
        data (np.ndarray): Input data points.
        labels (List or np.ndarray): List of class labels.
        num_prototypes (int or List[int]): Number of prototypes per class.

    Returns:
        np.ndarray: Prototypes initialized from data.
        np.ndarray: Corresponding class labels for prototypes.
    """
    eps = 1e-4

    assert data.ndim == 3, f"data should be a 3d array. But it is a {data.ndim} array."

    unique_classes = np.unique(labels)
    _, embedding_dim, subspace_dim = data.shape

    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    if isinstance(num_prototypes, int):
        num_prototypes = np.full(len(unique_classes), num_prototypes)

    total_prototypes = np.sum(num_prototypes)
    prototypes = np.zeros((total_prototypes, embedding_dim, subspace_dim))

    idx = 0
    for class_label, n in zip(unique_classes, num_prototypes):
        class_indices = np.where(labels == class_label)[0]
        selected_indices = np.random.choice(class_indices, n, replace=False)
        sampled_data = data[selected_indices] + eps * np.random.randn(n, embedding_dim, subspace_dim)
        prototypes[idx: idx + n] = np.array([orth(sample) for sample in sampled_data])
        idx += n

    prototype_labels = np.repeat(unique_classes, num_prototypes)

    return prototypes, prototype_labels

# --------------------------- Loss Functions ---------------------------

def sigmoid(x: float, sigma: int = 100) -> float:
    return 1 / (1 + np.exp(-sigma * x))


class SigmoidLoss:
    """
    Sigmoid-based loss function.
    """

    def __init__(self, sigma: int = 100):
        """
        Initializes the loss function with an optional scaling factor.

        Args:
            sigma (int, optional): Scaling factor for the sigmoid. Defaults to 100.
        """
        self.sigma = sigma

    def __call__(self, distance_plus: float, distance_minus: float) -> float:
        """
        Computes the sigmoid loss.

        Args:
            distance_plus (float): Distance to the prototype with the same label.
            distance_minus (float): Distance to the prototype with a different label.

        Returns:
            float: Calculated sigmoid-based loss.
        """
        score = (distance_plus - distance_minus) / (distance_plus + distance_minus)
        return sigmoid(score, sigma=self.sigma)


class IdentityLoss:
    """
    Identity-based loss function.
    """

    def __call__(self, distance_plus: float, distance_minus: float) -> float:
        """
        Computes the identity loss.

        Args:
            distance_plus (float): Distance to the prototype with the same label.
            distance_minus (float): Distance to the prototype with a different label.

        Returns:
            float: Calculated identity-based loss.
        """
        score = (distance_plus - distance_minus) / (distance_plus + distance_minus)
        return score


# --------------------------- Distance Calculations ---------------------------

def compute_distances(data: np.ndarray,
                      prototypes: np.ndarray,
                      relevance: np.ndarray) -> dict:
    """
    Calculate distances between data points and prototypes using geodesic or chordal distance.

    Args:
        data (np.ndarray): Input data.
        prototypes (np.ndarray): Prototype subspaces.
        relevance (np.ndarray, optional): Relevance (or lambda) factors.

    Returns:
        dict: Distances and SVD results.
    """
    assert data.shape[-2:] == prototypes.shape[-2:]
    assert relevance.shape[0] == 1
    if data.ndim == 2:
        data = data[np.newaxis, np.newaxis, :]
    elif data.ndim == 3:
        data = data[:, np.newaxis, :]

    U, S, Vh = np.linalg.svd(np.transpose(data, (0, 1, 3, 2)) @ prototypes,
                             full_matrices=False, compute_uv=True, hermitian=False)

    distances = 1 - np.transpose(relevance @ np.transpose(S, (0, 2, 1)), (0, 2, 1))

    return {
        'Q': np.squeeze(U),
        'Qw': np.squeeze(np.transpose(Vh, (0, 1, 3, 2))), #np.squeeze(Vh),
        'canonical_corr': np.squeeze(S),
        'distance': np.squeeze(distances)
    }


def find_closest_prototypes_indices(distances: np.ndarray,
                            label: int,
                            prototype_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find nearest prototypes with the same and different labels.

    Args:
        distances (np.ndarray): Calculated distances.
        label (int): Class label.
        prototype_labels (np.ndarray): Labels of prototypes.

    Returns:
        tuple: Indices of nearest same-class and different-class prototypes.
    """

    same_class_indices = np.where(prototype_labels == label)[0]
    diff_class_indices = np.where(prototype_labels != label)[0]

    nearest_same = same_class_indices[np.argmin(distances[same_class_indices])]
    nearest_diff = diff_class_indices[np.argmin(distances[diff_class_indices])]

    return nearest_same, nearest_diff

def find_closest_prototypes(
    data: np.ndarray,
    label: int,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray,
    relevance: np.ndarray
) -> Tuple[dict, dict]:
    """
    Identify the closest prototypes to a given data point.

    Args:
        data (np.ndarray): Input data point.
        label (int): Class label of the data point.
        prototypes (np.ndarray): Prototype subspaces.
        prototype_labels (np.ndarray): Labels corresponding to prototypes.
        relevance (np.ndarray): Relevance matrix.

    Returns:
        dict: Closest prototype with the same label.
        dict: Closest prototype with a different label.
    """
    distance_results = compute_distances(data, prototypes, relevance)

    nearest_same_index, nearest_diff_index = find_closest_prototypes_indices(distance_results['distance'],
                                                                             label, prototype_labels)
    distances = distance_results['distance']

    def create_prototype_dict(index):
        return {
            'index': index,
            'distance': distances[index],
            'Q': distance_results['Q'][index],
            'Qw': distance_results['Qw'][index],
            'canonical_corr': distance_results['canonical_corr'][index]
        }

    closest_same = create_prototype_dict(nearest_same_index)
    closest_diff = create_prototype_dict(nearest_diff_index)

    return closest_same, closest_diff



# --------------------------- Visualization ---------------------------

def evaluate_model_performance(targets: np.ndarray, predictions: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Evaluate model performance by calculating accuracy and confusion matrix.

    Args:
        targets (np.ndarray): Ground truth labels.
        predictions (np.ndarray): Predicted labels.

    Returns:
        Tuple[float, np.ndarray]: Accuracy percentage and confusion matrix.
    """
    assert targets.shape == predictions.shape, (
        f'Shape mismatch: targets shape {targets.shape}, predictions shape {predictions.shape}'
    )

    accuracy = accuracy_score(targets, predictions) * 100
    conf_matrix = confusion_matrix(targets, predictions, normalize=None)

    return accuracy, conf_matrix

# --------------------------- Visualization ---------------------------

def plot_accuracy_curve(acc_train: List[float],
                        acc_val: List[float] = None,
                        save_path: str = "") -> None:
    """
    Plot training and validation accuracy curves.

    Args:
        acc_train (List[float]): Training accuracy values over epochs.
        acc_val (List[float], optional): Validation accuracy values over epochs. Defaults to None.
        save_path (str, optional): The path to save the plot. Defaults to "".
    """
    nepochs = list(range(len(acc_train)))
    plt.figure(figsize=(8, 4))
    plt.plot(nepochs, acc_train, label='Train Set')

    if acc_val is not None:
        plt.plot(nepochs, acc_val, label='Validation Set')

    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='eps', dpi=150)
    plt.show()


def plot_relevance_factors(relevances: np.ndarray, save_path: str = "") -> None:
    """
    Plot relevance factors as a bar chart.

    Args:
        relevances (np.ndarray): Relevance values to plot.
        save_path (str, optional): the path for saving the plot. Defaults to "".
    """
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.barplot(x=list(range(1, relevances.shape[0] + 1)), y=relevances, palette="Greens", edgecolor=".1", ax=ax)

    ax.set_xlabel("Index of Relevance Factors", fontweight="bold", fontsize=9)
    ax.set_ylabel("Relevance Factors", fontweight="bold", fontsize=9)
    ax.set_yticks([0, 0.1, 0.2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="eps", dpi=150)

    plt.show()

def save_model(model, filepath: str) -> None:
    """
    Save the trained model to a file.

    Args:
    - filepath (str): Path where the model will be saved.
    """
    dir_name = "/".join(filepath.split("/")[:-1])
    os.makedirs(dir_name, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath: str):
    """
    Load a trained model from a file.

    Args:
    - filepath (str): Path to the saved model file.
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
