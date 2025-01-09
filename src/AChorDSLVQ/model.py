import numpy as np

from src.utils.utils import (
    initialize_random_prototypes, initialize_prototypes_from_data,
    compute_distances,
)

class AChorDSLVQModel:
    """
    A model class for training a prototype-based classifier using distance metrics
    such as geodesic and chordal distances. This model handles prototype initialization,
    forward pass, and loss computation.

    Attributes:
    - embedding_dim (int): Dimensionality of the word-embedding model.
    - subspace_dim (int): Dimensionality of the embedding subspace.
    - nprotos (int): Number of prototypes.
    - sigma (float): Scaling factor for the sigmoid activation function.
    - metric_type (str): The type of metric ('chordal').
    - num_of_classes (int): Number of classes for classification.
    - act_fun (str): The activation function used ('sigmoid', 'identity', etc.).
    """
    def __init__(self, embedding_dim, subspace_dim, **kwargs):
        """
        Initializes the model with the given data dimensions and hyperparameters.

        Args:
        - embedding_dim (int): Dimensionality of the input data.
        - subspace_dim (int): Dimensionality of the subspace.
        - kwargs (dict): Additional hyperparameters such as maxepochs, metric_type, etc.
        """

        self.embedding_dim = embedding_dim
        self.subspace_dim = subspace_dim
        self.metric_type = 'chordal'

        self.num_epochs = kwargs.get('num_epochs', 100)
        self.num_prototypes = kwargs.get('num_prototypes', 1)
        self.scaling_factor_sigmoid = kwargs.get('sigma', 100)
        self.activation_function = kwargs.get('act_fun', 'sigmoid')
        self.num_classes = kwargs.get('num_classes', 2)
        self.initialization_type = kwargs.get('prototype_init_type', 'random')

        self.low_bound_lambda = 1e-4
        self.learning_rate_weights = kwargs.get("learning_rate_prototypes", 0.01)
        self.learning_rate_relevance = kwargs.get('learning_rate_lambda', 0.0001)

        self.prototype_features, self.prototype_labels = initialize_random_prototypes(self.embedding_dim,
                                                                                      self.subspace_dim,
                                                                                      self.num_classes,
                                                                                      self.num_prototypes)
        self.lambda_value = np.ones(
            (1, self.prototype_features.shape[-1])
        ) / self.prototype_features.shape[-1]
        self.initial_prototypes = np.copy(self.prototype_features)
        self.initial_relevance_values = np.copy(self.lambda_value)


    def initialize_prototypes_with_samples(self, x_train, y_train) -> None:
        """
        Initializes the prototypes based on the provided training data.

        Args:
        - kwargs (dict): Contains training data or class information to initialize prototypes.
        """

        self.prototype_features, self.prototype_labels = initialize_prototypes_from_data(
            x_train,
            y_train,
            self.num_prototypes
        )
        self.initialization_type = 'samples'
        self.initial_prototypes = np.copy(self.prototype_features)

    def initialize_lambda_values(self, vector: np.ndarray):
        try:
            if vector.ndim == 1:
                assert len(vector) == self.lambda_value.shape[-1], (f"The provided vector should have "
                                                                    f"{self.lambda_value.shape[-1]} elements"
                                                                    f"but it has {len(vector)}.")
            elif vector.ndim == 2:
                assert vector.shape == self.lambda_value.shape, (f"The provided matrix should have "
                                                                f"{self.lambda_value.shape} shape"
                                                                f"but its shape is {vector.shape}.")
        except Exception as e:
            print(str(e))


    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Makes a prediction for the given input data based on the closest prototype.

        Args:
        - input_data (ndarray): The input data to classify.

        Returns:
        - ndarray: Predicted labels for the input data.
        """
        # Compute the distances from the input data to prototypes
        results = compute_distances(input_data, self.prototype_features, self.lambda_value)
        distances = results['distance']

        # If the input is 2D, distances will have shape (n_samples, n_prototypes)
        # For 1D input, distances is a vector, so we handle that case separately
        if distances.ndim == 1:  # Single data point case
            return self.prototype_labels[np.argmin(distances)]
        else:  # Multiple data points case
            # Find the indices of the closest prototypes for each data point
            winners = np.argmin(distances, axis=1)
            predictions = self.prototype_labels[winners]

        return predictions


