import logging
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array, check_is_fitted

from spn.algorithms.LearningWrappers import learn_classifier, learn_parametric
from spn.algorithms.MPE import mpe
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.gpu.TensorFlow import optimize_tf
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian, Parametric

logger = logging.getLogger(__name__)


class SPNClassifier(BaseEstimator, ClassifierMixin):
    """
    SPNClassifier wraps the SPN structure learning, tensorflow weight optimization and MPE procedures into a single
    class that follows the sklearn estimator interace. Therefore, SPNClassifier is usable in the sklearn framework as
    estimator in cross_val_score, GridSearchCV and more.
    """

    def __init__(
        self,
        parametric_types: List[Parametric] = None,
        n_jobs=-1,
        tf_optimize_weights=False,
        tf_n_epochs=100,
        tf_batch_size: int = None,
        tf_optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
        tf_pre_optimization_hook=None,
        tf_post_optimization_hook=None,
        tf_epoch_hook=None,
        min_instances_slice=50,
    ):
        """
        Create an SPNClassifier.

        :param parametric_types: Parametric types of leaf nodes. If None, all are assumed to be Gaussian
        n_jobs: Number of parallel jobs for learning the SPN structure
        :param tf_optimize_weights: Optimize weights in tensorflow
        :param tf_n_epochs: Number of tensorflow optimization epochs
        :param tf_batch_size: Batch size for tensorflow optimization
        :param tf_optimizer: Tensorflow optimizer to use for optimization
        :param tf_pre_optimization_hook: Hook that takes an SPN and returns an SPN before the optimization step
        :param tf_post_optimization_hook: Hook that takes an SPN and returns an SPN after the optimization step
        """
        self.n_jobs = n_jobs
        self.tf_optimize_weights = tf_optimize_weights
        self.tf_n_epochs = tf_n_epochs
        self.tf_optimizer = tf_optimizer
        self.tf_batch_size = tf_batch_size
        self.parametric_types = parametric_types
        self.tf_pre_optimization_hook = tf_pre_optimization_hook
        self.tf_post_optimization_hook = tf_post_optimization_hook
        self.min_instances_slice = min_instances_slice
        self.tf_epoch_hook = tf_epoch_hook

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=True)

        # Merge X and y
        train_data = np.c_[X, y].astype(np.float32)

        # If no parametric types were given: Assumen that all leafs are gaussian
        if self.parametric_types is None:
            parametric_types = [Gaussian] * X.shape[1] + [Categorical]
        else:
            parametric_types = self.parametric_types

        # Learn classifier
        self._spn = learn_classifier(
            train_data,
            ds_context=Context(parametric_types=parametric_types).add_domains(train_data),
            spn_learn_wrapper=learn_parametric,
            label_idx=X.shape[1],
            cpus=self.n_jobs,
            min_instances_slice=self.min_instances_slice,
        )

        # Obtain stats
        self.stats = get_structure_stats_dict(self._spn)

        # If pre optimization hook has been defined, run now
        if self.tf_pre_optimization_hook:
            self._spn = self.tf_pre_optimization_hook(self._spn)

        # If optimization flag is set: optimize weights in tf
        if self.tf_optimize_weights:
            self._spn, self.loss = optimize_tf(
                spn=self._spn,
                data=train_data,
                optimizer=self.tf_optimizer,
                batch_size=self.tf_batch_size,
                epochs=self.tf_n_epochs,
                return_loss=True,
                epoch_hook=self.tf_epoch_hook,
            )

        # If post optimization hook has been defined, run now
        if self.tf_post_optimization_hook:
            self._spn = self.tf_post_optimization_hook(self._spn)

        # # Compute marginalized spn
        # self._spn_marg = marginalize(self._spn, [X.shape[1]])

        self.X_ = X
        self.y_ = y

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = check_array(X)

        # Classify
        n_test = X.shape[0]
        y_empty = np.full((n_test, 1), fill_value=np.nan)
        data = np.c_[X, y_empty]
        data_filled = mpe(self._spn, data)
        y_pred = data_filled[:, -1]

        return y_pred

    # def predict_proba(self, X):
    #     # Check is fit had been called
    #     check_is_fitted(self, ["X_", "y_"])

    #     # Input validation
    #     X = check_array(X)

    #     # Classify
    #     n_test = X.shape[0]
    #     y_0 = np.full((n_test, 1), fill_value=0)
    #     y_1 = np.full((n_test, 1), fill_value=1)
    #     data_0 = np.c_[X, y_0]
    #     data_1 = np.c_[X, y_1]
    #     data_0 = data_0.astype(np.float64)
    #     data_1 = data_1.astype(np.float64)

    #     ll_0 = np.exp(log_likelihood(self._spn, data_0))
    #     ll_1 = np.exp(log_likelihood(self._spn, data_1))

    #     y_probas = np.c_[ll_0, ll_1]
    #     y_probas /= np.sum(y_probas, axis=1)[:, np.newaxis]
    #     return y_probas

    def get_params(self, deep=True):
        """Method to make SPNClassifier usable in sklearn procedures such as cross_val_score etc."""
        return {
            "parametric_types": self.parametric_types,
            "n_jobs": self.n_jobs,
            "tf_optimize_weights": self.tf_optimize_weights,
            "tf_n_epochs": self.tf_n_epochs,
            "tf_batch_size": self.tf_batch_size,
            "tf_optimizer": self.tf_optimizer,
            "tf_pre_optimization_hook": self.tf_pre_optimization_hook,
            "tf_post_optimization_hook": self.tf_post_optimization_hook,
        }

    def set_params(self, **parameters):
        """Method to make SPNClassifier usable in sklearn procedures such as cross_val_score etc."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def classification_categorical_to_tf_graph(
    node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32
):
    """
    Fix categorical to tf graph for classification problem.

    For a binary class label, there will be two categorical leaf nodes in the SPN. One which one-hot encodes the first
    class as [0, 1] and one that encodes the second clas as [1, 0]. 
    
    Since the tf optimizes the log likelihood, these one-hot represented probabilities will be projected into logspace
    which results in log([1,0])=[0, -inf] and therefore NaNs in further computations.

    Therefore, this custom method adds a small epsilon, such that the zero probability value in the one-hot vector will
    not degrade to negative infinity.
    """
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        p = np.array(node.p, dtype=dtype)

        # Epsilon to make sure there are no zero values
        eps = 1e-9
        p += eps

        # Renormalize such that the sum over all probabilities is one
        p /= np.sum(p)

        assert np.all(p > 0), "Probabilities in the class leaf nodes have to be greater than zero but were %s" % p

        softmaxInverse = np.log(p / np.max(p)).astype(dtype)
        probs = tf.nn.softmax(tf.constant(softmaxInverse))
        variable_dict[node] = probs
        if log_space:
            return tf.distributions.Categorical(probs=probs).log_prob(data_placeholder[:, node.scope[0]])

        return tf.distributions.Categorical(probs=probs).prob(data_placeholder[:, node.scope[0]])
