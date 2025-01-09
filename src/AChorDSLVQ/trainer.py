import numpy as np
from scipy import linalg as LA
from tqdm import tqdm
from src.AChorDSLVQ.model import AChorDSLVQModel
from src.utils.utils import evaluate_model_performance, find_closest_prototypes
from src.utils.utils import SigmoidLoss, IdentityLoss
from src.AChorDSLVQ.grad_utils import (
    ChordalDistanceDerivativeForPrototypes, ChordalDistanceRelevanceDerivative, MuFunctionDerivativePLUS,
    MuFunctionDerivativeMINUS, SigmoidLossDerivative, IdentityLossDerivative
)


class ModelTrainer:
    def __init__(self,
                 classifier: AChorDSLVQModel,
                 params: dict):
        """
        Initializes the model trainer with the given model and gradient computation class.

        Args:
            classifier: The model to be trained.
            params: Input parameters for the classifier
        """
        super().__init__()
        self.model = classifier
        self.loss = SigmoidLoss(sigma=params['sigma']) if params['act_fun'] == 'sigmoid' else IdentityLoss()
        self.loss_derivative = SigmoidLossDerivative(sigma=params['sigma']) if params['act_fun'] == 'sigmoid' else IdentityLossDerivative()

        self.mu_derivative_plus = MuFunctionDerivativePLUS()
        self.mu_derivative_minus = MuFunctionDerivativeMINUS()
        self.prototype_derivative = ChordalDistanceDerivativeForPrototypes()
        self.relevance_derivative = ChordalDistanceRelevanceDerivative()

    def _update_prototypes_and_relevance(self, data, label):
        plus, minus = find_closest_prototypes(data, label,
                                              self.model.prototype_features,
                                              self.model.prototype_labels,
                                              self.model.lambda_value)
        cost = self.loss(plus['distance'], minus['distance'])

        # Rotate the coordinate system
        data_rot_plus = data @ plus['Q']
        data_rot_minus = data @ minus['Q']
        proto_rot_plus = self.model.prototype_features[plus['index']] @ plus['Qw']
        proto_rot_minus = self.model.prototype_features[minus['index']] @ minus['Qw']

        # Computation of gradients
        gradient_loss = self.loss_derivative(cost)
        gradient_mu_plus = self.mu_derivative_plus(plus['distance'], minus['distance'])
        gradient_mu_minus = self.mu_derivative_minus(plus['distance'], minus['distance'])
        gradient_prototype_plus = self.prototype_derivative(self.model.lambda_value, data_rot_plus)
        gradient_prototype_minus = self.prototype_derivative(self.model.lambda_value, data_rot_minus)
        gradient_rel_plus = self.relevance_derivative(plus['canonical_corr'])
        gradient_rel_minus = self.relevance_derivative(minus['canonical_corr'])

        gradient_plus = gradient_loss * gradient_mu_plus * gradient_prototype_plus
        gradient_minus = gradient_loss * gradient_mu_minus * gradient_prototype_minus
        gradient_lambda =  gradient_loss * (gradient_mu_plus * gradient_rel_plus +
                                            gradient_mu_minus * gradient_rel_minus )

        # update winner prototypes
        self.model.prototype_features[plus['index']] = proto_rot_plus - self.lr_w * gradient_plus
        self.model.prototype_features[minus['index']] = proto_rot_minus - self.lr_w * gradient_minus

        # Orthonormalize prototypes
        self.model.prototype_features[plus['index']] = LA.orth(self.model.prototype_features[plus['index']])
        self.model.prototype_features[minus['index']] = LA.orth(self.model.prototype_features[minus['index']])

        # Update relevance factors
        self.model.lambda_value[0] -= self.lr_r * gradient_lambda

        # Normalize relevance factors
        self.model.lambda_value[
            0, np.argwhere(self.model.lambda_value < self.model.low_bound_lambda)[:, 1]] = self.model.low_bound_lambda
        self.model.lambda_value[0] = self.model.lambda_value[0] / np.sum(self.model.lambda_value)

        return cost

    def _train_one_epoch(self, xtrain, ytrain, **kwargs):
        """
        Perform one epoch of training using the provided training data.
        """
        # Set learning rates
        self.lr_w = kwargs.get('lr_w', self.model.learning_rate_weights)
        self.lr_r = kwargs.get('lr_r', self.model.learning_rate_relevance)

        total_cost = 0

        # Shuffle data for stochastic gradient descent
        perm = np.random.permutation(xtrain.shape[0])
        for data, label in tqdm(zip(xtrain[perm], ytrain[perm]), total=len(ytrain)):
            total_cost += self._update_prototypes_and_relevance(data, label)

        return total_cost/xtrain.shape[0]


    def train(self,
              xtrain, ytrain,
              xval, yval,
              logger, **kwargs):
        """
        Train the model for a specified number of epochs.

        Args:
            xtrain (ndarray): Training data.
            ytrain (ndarray): Training labels.
            xval (ndarray): Validation data.
            yval (ndarray): Validation labels.
            logger (logging.Logger): logger.
            **kwargs: Additional parameters for training (learning rates, bounds, etc.).
        """
        nepochs = self.model.num_epochs

        for epoch in range(nepochs):
            logger.info(f"Epoch {epoch + 1}/{nepochs}")
            avg_cost = self._train_one_epoch(xtrain, ytrain, **kwargs)
            logger.info(f"Average Training Cost: {avg_cost}")

            ytrain_pred = self.model.predict(xtrain)
            acc, conf_matrix = evaluate_model_performance(ytrain, ytrain_pred)

            logger.info(f"Training Accuracy: {acc}")
            print("Confusion Matrix: \n", conf_matrix)

            if xval is not None:
                yval_pred = self.model.predict(xval)
                acc_val, conf_matrix_val = evaluate_model_performance(yval, yval_pred)

                logger.info(f"Validation Accuracy: {acc_val}")


