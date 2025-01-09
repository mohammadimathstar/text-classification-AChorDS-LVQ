import numpy as np

class IdentityLossDerivative:
    """
    Derivative of the Identity-based loss function.
    """

    def __call__(self, input_value: float) -> float:
        """
        Computes the derivative of the identity loss function with respect to its input.

        Args:
            input_value (float): the input value to the identity function

        Returns:
            float: Derivative of the identity loss.
        """
        # The derivative of the identity loss is always 1

        return 1


class SigmoidLossDerivative:
    """
    Derivative of the Sigmoid-based loss function.
    """

    def __init__(self, sigma: int = 100):
        """
        Initializes the derivative of the loss function with an optional scaling factor.

        Args:
            sigma (int, optional): Scaling factor for the sigmoid. Defaults to 100.
        """
        self.sigma = sigma

    def __call__(self, input_value: float) -> float:
        """
        Computes the derivative of the sigmoid loss function with respect to its input.

        Args:
            input_value (float): the input value to the sigmoid value

        Returns:
            float: Derivative of the sigmoid loss.
        """

        # Compute the derivative of the sigmoid activation
        derivative = self.sigma * input_value * (1 - input_value)

        return derivative


class LossDerivative:
    def __init__(self, activation_function, sigma=100):
        if activation_function == 'sigmoid':
            self.gradient_activation_function = SigmoidLossDerivative(sigma=sigma)
        else:
            self.gradient_activation_function = IdentityLossDerivative()

    def __call__(self, input_value):
        return self.gradient_activation_function(input_value)


class MuFunctionDerivativePLUS:
    """
    A class to compute the derivative of the mu function with respect to either
    dplus or dminus, where mu is defined as:

    mu = (dplus - dminus) / (dplus + dminus)

    Attributes:
    - is_derivative_wrt_dplus: A boolean flag to compute the derivative with respect to dplus (True)
                               or dminus (False).
    """

    def __call__(self, dplus: float, dminus: float) -> float:
        """
        Computes the derivative of the mu function with respect to either dplus or dminus.

        The formula for the mu function is:

        mu = (dplus - dminus) / (dplus + dminus)

        Args:
        - dplus (float): The distance to the positive prototype.
        - dminus (float): The distance to the negative prototype.

        Returns:
        - float: The computed derivative of the mu function.
        """

        return 2 * dminus / ((dplus + dminus) ** 2)



class MuFunctionDerivativeMINUS:
    """
    A class to compute the derivative of the mu function with respect to either
    dplus or dminus, where mu is defined as:

    mu = (dplus - dminus) / (dplus + dminus)

    Attributes:
    - is_derivative_wrt_dplus: A boolean flag to compute the derivative with respect to dplus (True)
                               or dminus (False).
    """

    def __call__(self, dplus: float, dminus: float) -> float:
        """
        Computes the derivative of the mu function with respect to either dplus or dminus.

        The formula for the mu function is:

        mu = (dplus - dminus) / (dplus + dminus)

        Args:
        - dplus (float): The distance to the positive prototype.
        - dminus (float): The distance to the negative prototype.

        Returns:
        - float: The computed derivative of the mu function.
        """

        # Derivative with respect to dminus
        return -2 * dplus / ((dplus + dminus) ** 2)


class ChordalDistanceDerivativeForPrototypes:
    """
        Class to compute the gradient (derivative) of the chordal distance with respect to the winner prototypes.
        This derivative is used during the training process when adjusting the prototypes.
    """

    def __call__(self, lambda_values: np.ndarray, rotated_data: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the chordal distance with respect to the winner prototypes.

        Args:
        - rotated_data (ndarray): The rotated data.

        Returns:
        - ndarray: The gradient of the chordal distance with respect to the winner prototype.
        """
        embedding_dim = rotated_data.shape[-2] # TODO: it has been changed (lambda_vales.shape[-2])
        L = np.tile(lambda_values[0], (embedding_dim, 1))
        return - L * rotated_data


class ChordalDistanceRelevanceDerivative:
    """
        Class to compute the gradient (derivative) of the chordal distance with respect to the relevance vector (lambda).
        This derivative is used during the training process when adjusting the relevance factors.
    """

    def __call__(self, canonical_correlation: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the chordal distance with respect to the lambda (relevance) vector.

        Args:
        - canonical_correlation (ndarray): canonical correlation

        Returns:
        - ndarray: The gradient of the chordal distance with respect to the lambda (relevance) vector.
        """
        return - canonical_correlation.T


