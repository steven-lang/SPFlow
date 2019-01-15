from typing import Tuple
import argparse
from functools import partial
import logging
import os
import sys
import math
import numpy as np

import maxout
import sklearn.datasets
import tensorflow as tf
from maxout import (
    Maxout,
    augment_spn_maxout_all,
    create_sum_to_maxout_func,
    sum_to_maxout_rand,
    sklearn_classifier_pre_opt_maxout as mohook,
)
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split, cross_validate
from spn.algorithms.sklearn import SPNClassifier
from spn.algorithms.LearningWrappers import learn_classifier, learn_parametric
from spn.gpu.TensorFlow import add_node_to_tf_graph, optimize_tf
from spn.io.plot.TreeVisualization import plot_spn
from spn.structure.Base import Context, Sum, assign_ids, get_nodes_by_type, rebuild_scopes_bottom_up
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.structure.leaves.parametric.Parametric import CategoricalDictionary

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(stream=sys.stdout), logging.FileHandler("log.txt", mode="a")],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# K test values
K_TEST_VALUES = [2, 3, 4, 5, 8, 16, 32]
K_DEFAULT = 4

# Number of Tensorflow optimization epochs
TF_N_EPOCHS = 100
TF_LEARNING_RATE_VALUES = [2 ** (-i) for i in range(1, 10)]
TF_LEARNING_RATE_DEFAULT = 0.01

# Number of synthetic samples
N_SYNTH_SAMPLES = 500
N_SYNTH_NFEAT_VALUES = [8, 64]
N_SYNTH_PINF_VALUES = [0.25, 1.0]

# Weight augmentation factors
WAF_FACTORS = [0.1, 0.3, 0.5]

logger = logging.getLogger(__name__)


def evaluate_cv(spn: SPNClassifier, X, y, n_splits=5) -> Tuple[float, float]:
    """
    Evaluate spn with cross validation.

    Parameters
    ----------
    spn : SPNClassifier
        Input classifier
    X : np.ndarray
        Data features
    y : np.ndarray
        Data targets
    n_splits : int
        Number of splits in cross validation.
    """
    # Prepare data
    scorer = make_scorer(roc_auc_score)
    n_jobs = 1
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    result = cross_validate(spn, X, y, n_jobs=n_jobs, verbose=0, cv=cv, scoring=scorer)
    return np.average(result["train_score"]), np.average(result["test_score"])


def my_categorical_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    """Fix categorical to tf graph"""
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        p = np.array(node.p, dtype=dtype)
        eps = 1e-20
        p += eps
        p /= np.sum(p)

        softmaxInverse = np.log(p / np.max(p)).astype(dtype)
        probs = tf.nn.softmax(tf.constant(softmaxInverse))
        variable_dict[node] = probs
        if log_space:
            return tf.distributions.Categorical(probs=probs).log_prob(data_placeholder[:, node.scope[0]])

        return tf.distributions.Categorical(probs=probs).prob(data_placeholder[:, node.scope[0]])


def load_audit():
    """Load the audit dataset"""
    data = np.loadtxt("audit_risk.csv", delimiter=",", comments="#")
    X = data[:, 0:-2]
    y = data[:, -1]
    return X, y


def run_exp_maxout(X, y, maxout_transform_func):
    """Experiment that tests maxout with pure random initialization"""
    n_epochs = TF_N_EPOCHS

    ks = []
    auc_score_train = []
    auc_score_test = []

    # Run SPN Vanilla without maxout as k=0
    spn_vanilla = SPNClassifier(tf_optimize_weights=True, tf_n_epochs=n_epochs)
    vanilla_score_train, vanilla_score_test = evaluate_cv(spn_vanilla, X, y)
    ks.append(0)
    auc_score_train.append(vanilla_score_train)
    auc_score_test.append(vanilla_score_test)

    for k in K_TEST_VALUES:
        logger.info("k={}".format(k))
        spn_maxout = SPNClassifier(
            tf_optimize_weights=True,
            tf_optimizer=tf.train.AdamOptimizer(learning_rate=TF_LEARNING_RATE_DEFAULT),
            tf_n_epochs=n_epochs,
            tf_pre_optimization_hook=mohook(maxout_k=k, transform_func=maxout_transform_func),
            min_instances_slice=int(X.shape[0]/10),
        )
        train_score, test_score = evaluate_cv(spn_maxout, X, y)
        ks.append(k)
        auc_score_train.append(train_score)
        auc_score_test.append(test_score)

    return ks, auc_score_train, auc_score_test


def run_exp_maxout_random_init(X, y):
    return run_exp_maxout(X, y, sum_to_maxout_rand)


def run_exp_maxout_from_sum_weights(X, y, weight_augmentation_factor):
    sum_to_maxout = create_sum_to_maxout_func(weight_augmentation_factor)
    return run_exp_maxout(X, y, sum_to_maxout)


def load_iris_3d():
    return load_iris(return_X_y=True)


def load_iris_2d():
    X, y = load_iris(return_X_y=True)
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]
    return X, y


def load_wine_3d():
    X, y = sklearn.datasets.load_wine(return_X_y=True)
    return X, y


def load_wine_2d():
    X, y = sklearn.datasets.load_wine(return_X_y=True)
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    return X, y


def store_results(dataset_name, exp_name, setup_name, column_names, data):
    results_dir = args.result_dir
    exp_dir = os.path.join(results_dir, exp_name)
    ensure_dir(exp_dir)
    # Write header and content into a csv file
    fname = "./{}/{}-{}.csv".format(exp_dir, dataset_name, setup_name)
    np.savetxt(fname, data, delimiter=",", header=",".join(column_names))


def generate_spn_plot(fname, X, y):
    spn = SPNClassifier(min_instances_slice=int(X.shape[0]/10))
    spn.fit(X, y)
    plot_spn(spn._spn, file_name=fname)


def run_on_data(dataset_name, X, y):
    logger.info("Starting experiments on %s", dataset_name)
    generate_spn_plot(fname=os.path.join(args.result_dir, "spn-{}.png".format(dataset_name)), X=X, y=y)

    def run_sum_to_maxout(weight_augmentation_factor):
        # Run sum weight transformation experiment
        ks, train_score, test_score = run_exp_maxout_from_sum_weights(X, y, weight_augmentation_factor)
        data = np.c_[ks, train_score, test_score]
        store_results(
            dataset_name=dataset_name,
            exp_name="auc-eval",
            setup_name="from-sum-weight-waf-{}".format(weight_augmentation_factor),
            column_names=["k", "AUC"],
            data=data,
        )

    # Run sum-to-maxout experiment with different weight augmentation factors
    for waf in WAF_FACTORS:
        run_sum_to_maxout(waf)

    # Run random init experiment
    ks, train_score, test_score = run_exp_maxout_random_init(X, y)
    data = np.c_[ks, train_score, test_score]
    store_results(
        dataset_name=dataset_name, exp_name="auc-eval", setup_name="random-init", column_names=["k", "AUC"], data=data
    )


def run_loss_experiments(dataset_name, X, y):
    """Run the loss comparison experiments"""

    def run(maxout_transform_func, tag):
        column_names = []
        losses = []

        def add_loss(spn, label):
            loss = spn.fit(X, y).loss
            column_names.append(label)
            losses.append(loss)

        # SPN without maxout
        spn = SPNClassifier(
            tf_optimize_weights=True,
            tf_n_epochs=TF_N_EPOCHS,
            tf_optimizer=tf.train.AdamOptimizer(learning_rate=TF_LEARNING_RATE_DEFAULT),
            min_instances_slice=int(X.shape[0]/10)
        )
        add_loss(spn, "maxout=False")

        # SPN with different maxout values (random init)
        for k in K_TEST_VALUES:
            spn = SPNClassifier(
                tf_optimize_weights=True,
                tf_n_epochs=TF_N_EPOCHS,
                tf_optimizer=tf.train.AdamOptimizer(learning_rate=TF_LEARNING_RATE_DEFAULT),
                tf_pre_optimization_hook=mohook(maxout_k=k, transform_func=maxout_transform_func),
            )
            add_loss(spn, "maxout=%s" % k)

        loss_data = np.array(losses).T
        data = np.c_[range(TF_N_EPOCHS), loss_data]
        column_names = ["epoch"] + column_names
        store_results(
            dataset_name=dataset_name, exp_name="loss-eval", setup_name=tag, column_names=column_names, data=data
        )

    run(sum_to_maxout_rand, tag="random-init")
    for waf in WAF_FACTORS:
        run(create_sum_to_maxout_func(weight_augmentation_factor=waf), tag="from-sum-weight-waf-{}".format(waf))


def run_learning_rate_experiments(dataset_name, X, y):
    """Run experiments with learning rates"""

    def run(maxout_transform_func, tag):
        column_names = []
        losses = []

        def add_loss(spn, label):
            loss = spn.fit(X, y).loss
            column_names.append(label)
            losses.append(loss)

        # SPN without maxout
        spn = SPNClassifier(
            tf_optimize_weights=True,
            tf_n_epochs=TF_N_EPOCHS,
            tf_optimizer=tf.train.AdamOptimizer(learning_rate=TF_LEARNING_RATE_DEFAULT),
            min_instances_slice=int(X.shape[0]/10)
        )
        add_loss(spn, "maxout=False")

        # SPN with different maxout values (random init)
        for learning_rate in TF_LEARNING_RATE_VALUES:
            if maxout_transform_func is None:
                hook = None
            else:
                hook = mohook(maxout_k=K_DEFAULT, transform_func=maxout_transform_func)

            spn = SPNClassifier(
                tf_optimize_weights=True,
                tf_n_epochs=TF_N_EPOCHS,
                tf_optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
                tf_pre_optimization_hook=hook,
                min_instances_slice=int(X.shape[0]/10)
            )
            add_loss(spn, "lr=%s" % learning_rate)

        loss_data = np.array(losses).T
        data = np.c_[range(TF_N_EPOCHS), loss_data]
        column_names = ["epoch"] + column_names
        store_results(
            dataset_name=dataset_name,
            exp_name="learning-rate-eval",
            setup_name=tag,
            column_names=column_names,
            data=data,
        )

    run(None, tag="maxout=False")
    run(sum_to_maxout_rand, tag="random-init")
    for waf in WAF_FACTORS:
        run(create_sum_to_maxout_func(weight_augmentation_factor=waf), tag="from-sum-weight-waf-{}".format(waf))


def main_dataset_experiments():
    logger.info("Running main dataset experiments ...")
    main_run_experiment(run_on_data)


def main_loss_experiments():
    logger.info("Running main loss experiments ...")
    main_run_experiment(run_loss_experiments)


def main_learning_rate_experiments():
    logger.info("Running main learning rate experiments ...")
    main_run_experiment(run_learning_rate_experiments)


def main_run_experiment(exp_method):
    # Synthetic datasets
    # Different number of features
    for n_feat in N_SYNTH_NFEAT_VALUES:
        # Different ratio of informative features
        for p_inf in N_SYNTH_PINF_VALUES:
            logger.info(" -- Synth: n_feat=%s, p_inf=%s", n_feat, p_inf)
            # Num informative features
            n_inf = math.ceil(n_feat * p_inf)
            # Num redundant features
            n_red = n_feat - n_inf
            X, y = sklearn.datasets.make_classification(
                n_samples=N_SYNTH_SAMPLES,
                n_features=n_feat,
                n_informative=n_inf,
                n_redundant=n_red,
                n_classes=2,
                n_clusters_per_class=2,
                class_sep=1.0,
                random_state=42,
            )
            exp_method(dataset_name="synth-nfeat={}-pinf={}".format(n_feat, p_inf), X=X, y=y)

    # Non sythetic datasets
    if not args.debug and not args.synth_only:
        # Iris 2D
        X, y = load_iris_2d()
        logger.info(" -- Iris 2d --")
        exp_method(dataset_name="iris-2d", X=X, y=y)

        # Iris 3D
        X, y = load_iris_3d()
        logger.info(" -- Iris 3d --")
        exp_method(dataset_name="iris-3d", X=X, y=y)

        # Wine 2D
        X, y = load_wine_2d()
        logger.info(" -- Wine 2d --")
        exp_method(dataset_name="wine-2d", X=X, y=y)

        # Wine 3D
        X, y = load_wine_3d()
        logger.info(" -- Wine 3d --")
        exp_method(dataset_name="wine-3d", X=X, y=y)

        # Audit
        X, y = load_audit()
        logger.info(" -- Audit --")
        exp_method(dataset_name="audit", X=X, y=y)



def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="SPFlow Maxout Experiments")
    parser.add_argument("--result-dir", default="results", help="path to the result directory", metavar="DIR")
    parser.add_argument("--n-synth", "-ns", default=100, type=int, help="Number of synthetic samples")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--synth-only", default=False, action="store_true")
    args = parser.parse_args()

    ensure_dir(args.result_dir)
    return args


def ensure_dir(d):
    """Ensure that a directory exists"""
    # Create result dir on the fly
    if not os.path.exists(d):
        os.makedirs(d)


if __name__ == "__main__":
    # Limit tf GPU usage
    np.random.seed(1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    args = parse_args()
    N_SYNTH_SAMPLES = args.n_synth

    # Reduce default values to something small when debugging
    if args.debug:
        K_TEST_VALUES = [2]
        TF_N_EPOCHS = 5
        TF_LEARNING_RATE_VALUES = [0.1]
        N_SYNTH_SAMPLES = 10
        N_SYNTH_NFEAT_VALUES = [8]
        N_SYNTH_PINF_VALUES = [1.0]
        WAF_FACTORS = [0.1]

    # Register custom categorical_to_tf_graph implementation
    add_node_to_tf_graph(Categorical, my_categorical_to_tf_graph)

    # Start the three main experiments
    # main_learning_rate_experiments()
    main_dataset_experiments()
    main_loss_experiments()
