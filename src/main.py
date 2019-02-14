import argparse
import time
import logging
import os
import sys
from functools import partial
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.utils import shuffle

from spn.algorithms.Inference import log_likelihood
from spn.algorithms.MPE import mpe
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.gpu.TensorFlow import optimize_tf
from spn.structure.Base import Context
import datasets
from maxout import augment_spn_maxout_all, create_sum_to_maxout_func
from maxout import sklearn_classifier_pre_opt_maxout as mohook
from maxout import sum_to_maxout_rand
from spn.algorithms.LearningWrappers import learn_classifier, learn_parametric
from spn.algorithms.sklearn import SPNClassifier
from spn.gpu.TensorFlow import add_node_to_tf_graph, optimize_tf
from spn.io.plot.TreeVisualization import plot_spn
from spn.structure.Base import Context, Node
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(stream=sys.stdout), logging.FileHandler("log.txt", mode="a")],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# K test values
K_TEST_VALUES = [2, 3, 4, 5, 8, 16, 32, 64]
K_DEFAULT = 4

# Structure learning
MIN_INSTANCES_SLICE = 25

# Number of Tensorflow optimization epochs
TF_N_EPOCHS = 100
TF_LEARNING_RATE_VALUES = [2 ** (-i) for i in range(1, 10)]

# Number of synthetic samples
N_SYNTH_SAMPLES = 200  # TODO: SET TO 500
N_SYNTH_NFEAT_VALUES = [8, 64]
N_SYNTH_PINF_VALUES = [0.25, 1.0]

# Weight augmentation factors
WAF_FACTORS = [0.1]

logger = logging.getLogger(__name__)


def evaluate_spn(X: np.ndarray, y: np.ndarray, n_epochs, learning_rate: float, maxout_k: int, maxout_init: str = None):
    """
    Args:
        X: Input variables.
        y: Target variables.
        n_epochs: Number of training epochs.
        learning_rate: Tensorflow learning rate.
        maxout_k: Maxout k value.

    Returns:
        Tuple[List[float], List[float], List[float]]: Tuple of loss, training and testing accuracy scores over the
        number of epochs. 
    """
    # Setup train/test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    train_data = np.c_[X_train, y_train].astype(np.float32)

    # Parametric type: All gaussian except target
    parametric_types = [Gaussian] * X.shape[1] + [Categorical]

    # Learn classifier
    spn = learn_classifier(
        train_data,
        ds_context=Context(parametric_types=parametric_types).add_domains(train_data),
        spn_learn_wrapper=partial(learn_parametric, min_instances_slice=max(25, X_train.shape[0] / 10)),
        label_idx=X.shape[1],
        cpus=-1,
    )

    # Run maxout augmentation
    if maxout_k > 0:
        # Detect sum to maxout transformation function
        if maxout_init == "random":
            transform_func = sum_to_maxout_rand
        elif maxout_init == "augment":
            transform_func = create_sum_to_maxout_func(0.1)
        else:
            raise Exception("Invalid maxout init: %s" % maxout_init)

        # Apply transformation
        spn = augment_spn_maxout_all(spn, maxout_k, transform_func)

    def predict(spn, X_new):
        """Generate a prediction"""
        # Classify
        n_test = X_new.shape[0]
        y_empty = np.full((n_test, 1), fill_value=np.nan)
        data = np.c_[X_new, y_empty]
        data_filled = mpe(spn, data)
        y_pred = data_filled[:, -1]
        return y_pred

    # Store results collected during training
    loss_train_list = []
    loss_test_list = []
    acc_train_list = []
    acc_test_list = []

    def epoch_hook(epoch: int, train_loss: float, spn: Node):
        """Epoch hook that collects ACC score on train and test set"""
        # Compute test loss 
        test_loss = -1 *np.mean(log_likelihood(spn, np.c_[X_test, y_test]))

        # Get preds and probs
        y_train_pred = predict(spn, X_train)
        y_test_pred = predict(spn, X_test)

        # Get AUC and accuracy
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)

        # Store
        loss_train_list.append(train_loss)
        loss_test_list.append(test_loss)
        acc_train_list.append(acc_train)
        acc_test_list.append(acc_test)

    # Optimize SPN in tensorflow
    spn = optimize_tf(
        spn=spn,
        data=train_data,
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        batch_size=None,
        epochs=n_epochs,
        epoch_hook=epoch_hook,
    )

    return loss_train_list, loss_test_list, acc_train_list, acc_test_list


def time_delta_now(ts: float) -> str:
    """
    Convert a timestamp into a human readable timestring (%H:%M:%S).
    Args:
        ts (float): Timestamp

    Returns:
        Human readable timestring
    """
    a = ts
    b = time.time()  # current epoch time
    c = b - a  # seconds
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"


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
        eps = 1e-10
        p += eps
        p /= np.sum(p)

        softmaxInverse = np.log(p / np.max(p)).astype(dtype)
        probs = tf.nn.softmax(tf.constant(softmaxInverse))
        variable_dict[node] = probs
        if log_space:
            return tf.distributions.Categorical(probs=probs).log_prob(data_placeholder[:, node.scope[0]])

        return tf.distributions.Categorical(probs=probs).prob(data_placeholder[:, node.scope[0]])


def run_exp_maxout(X, y, maxout_transform_func):
    """Experiment that tests maxout with pure random initialization"""
    n_epochs = TF_N_EPOCHS

    ks = []
    auc_score_train = []
    auc_score_test = []

    # Run SPN Vanilla without maxout as k=0
    spn_vanilla = SPNClassifier(
        tf_optimize_weights=True,
        tf_n_epochs=n_epochs,
        tf_optimizer=tf.train.AdamOptimizer(learning_rate=args.learning_rate),
    )
    vanilla_score_train, vanilla_score_test = evaluate_cv(spn_vanilla, X, y)
    ks.append(0)
    auc_score_train.append(vanilla_score_train)
    auc_score_test.append(vanilla_score_test)

    for k in K_TEST_VALUES:
        logger.info("k={}".format(k))
        spn_maxout = SPNClassifier(
            tf_optimize_weights=True,
            tf_optimizer=tf.train.AdamOptimizer(learning_rate=args.learning_rate),
            tf_n_epochs=n_epochs,
            tf_pre_optimization_hook=mohook(maxout_k=k, transform_func=maxout_transform_func),
            min_instances_slice=MIN_INSTANCES_SLICE,
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


def store_results(dataset_name, exp_name, setup_name, column_names, data):
    results_dir = args.result_dir
    exp_dir = os.path.join(results_dir, exp_name)
    ensure_dir(exp_dir)
    # Write header and content into a csv file
    fname = "./{}/{}-{}.csv".format(exp_dir, dataset_name, setup_name)
    np.savetxt(fname, data, delimiter=",", header=",".join(column_names))


def gen_spn_plot_and_stats(fname, X, y):
    if not args.only_plots:
        return
    # Setup train/test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    train_data = np.c_[X_train, y_train].astype(np.float32)

    # Parametric type: All gaussian except target
    parametric_types = [Gaussian] * X.shape[1] + [Categorical]

    # Learn classifier
    spn = learn_classifier(
        train_data,
        ds_context=Context(parametric_types=parametric_types).add_domains(train_data),
        spn_learn_wrapper=partial(learn_parametric, min_instances_slice=max(25, X_train.shape[0] / 10)),
        label_idx=X.shape[1],
        cpus=-1,
    )

    stats = get_structure_stats_dict(spn)
    with open("{}-stats.txt".format(fname), "w") as f:
        # Print layers, nodes, edges
        print(stats)
        for k in ["layers", "nodes", "edges"]:
            f.write("{: <14}= {: >4}\n".format(k, stats[k]))

        # Print counts per type
        for k, v in stats["count_per_type"].items():
            f.write("{: <14}= {: >4}\n".format(k.__name__, v))

    plot_spn(spn, file_name="{}-plot.pdf".format(fname))


def run_on_data(dataset_name, X, y):
    logger.info("Starting experiments on %s", dataset_name)
    gen_spn_plot_and_stats(fname=os.path.join(args.result_dir, "spn-{}".format(dataset_name)), X=X, y=y)
    return

    def run_sum_to_maxout(weight_augmentation_factor):
        # Run sum weight transformation experiment
        ks, train_score, test_score = run_exp_maxout_from_sum_weights(X, y, weight_augmentation_factor)
        data = np.c_[ks, train_score, test_score]
        store_results(
            dataset_name=dataset_name,
            exp_name="auc-eval",
            setup_name="from-sum-weight-waf-{}".format(weight_augmentation_factor),
            column_names=["k", "auc train", "auc test"],
            data=data,
        )

    # Run sum-to-maxout experiment with different weight augmentation factors
    for waf in WAF_FACTORS:
        run_sum_to_maxout(waf)

    # Run random init experiment
    ks, train_score, test_score = run_exp_maxout_random_init(X, y)
    data = np.c_[ks, train_score, test_score]
    store_results(
        dataset_name=dataset_name,
        exp_name="auc-eval",
        setup_name="random-init",
        column_names=["k", "auc train", "auc test"],
        data=data,
    )


def run_loss_experiments(dataset_name, X, y):
    """Run the loss comparison experiments"""
    # Generate spn plots and stats
    d = os.path.join("plots", args.result_dir)
    ensure_dir(d)
    gen_spn_plot_and_stats(fname=os.path.join(d, "spn-{}".format(dataset_name)), X=X, y=y)

    # Skip the rest if only plots were to be made
    if args.only_plots:
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def run(maxout_init, tag):
        column_names = []
        loss_train_list = []
        loss_test_list = []
        acc_train_list = []
        acc_test_list = []

        def collect_result(loss_train, loss_test, acc_train, acc_test, label):
            column_names.append(label)
            loss_train_list.append(loss_train)
            loss_test_list.append(loss_test)
            acc_train_list.append(acc_train)
            acc_test_list.append(acc_test)

        # SPN without maxout
        loss_train, loss_test, acc_train, acc_test = evaluate_spn(
            X, y, n_epochs=TF_N_EPOCHS, learning_rate=args.learning_rate, maxout_k=0
        )
        collect_result(loss_train, loss_test, acc_train, acc_test, "maxout=False")

        # SPN with different maxout values (random init)
        for k in K_TEST_VALUES:
            loss_train, loss_test, acc_train, acc_test = evaluate_spn(
                X, y, n_epochs=TF_N_EPOCHS, learning_rate=args.learning_rate, maxout_k=k, maxout_init=maxout_init
            )
            collect_result(loss_train, loss_test, acc_train, acc_test, "maxout=%s" % k)

        # Collect auc data
        acc_train_data = np.c_[range(TF_N_EPOCHS), np.array(acc_train_list).T]
        acc_test_data = np.c_[range(TF_N_EPOCHS), np.array(acc_test_list).T]

        # Collect loss data
        loss_train_data = np.c_[range(TF_N_EPOCHS), np.array(loss_train_list).T]
        loss_test_data = np.c_[range(TF_N_EPOCHS), np.array(loss_test_list).T]
        column_names = ["epoch"] + column_names

        # Store acc results
        store_results(
            dataset_name=dataset_name,
            exp_name="acc-epoch-eval",
            setup_name=tag + "-train",
            column_names=column_names,
            data=acc_train_data,
        )

        store_results(
            dataset_name=dataset_name,
            exp_name="acc-epoch-eval",
            setup_name=tag + "-test",
            column_names=column_names,
            data=acc_test_data,
        )

        store_results(
            dataset_name=dataset_name,
            exp_name="loss-epoch-eval",
            setup_name=tag + "-train",
            column_names=column_names,
            data=loss_train_data,
        )

        store_results(
            dataset_name=dataset_name,
            exp_name="loss-epoch-eval",
            setup_name=tag + "-test",
            column_names=column_names,
            data=loss_test_data,
        )

    # Run random init experiment
    run("random", tag="random-init")
    run("augment", tag="augment-sum-init")


def main_dataset_experiments():
    logger.info("Running main dataset experiments ...")
    main_run_experiment(run_on_data)


def main_loss_experiments():
    logger.info("Running main loss experiments ...")
    main_run_experiment(run_loss_experiments)


def main_run_experiment(exp_method):
    dss = datasets.load_dataset_map()
    name = args.dataset
    loader = dss[name]
    X, y = loader()

    # If debug: shuffle X, y and only take the first 20 elements
    if args.debug:
        X, y = shuffle(X, y)
        X, y = X[:20], y[:20]
    logger.info(" -- %s -- Shape: %s", name, X.shape)
    t = time.time()
    exp_method(dataset_name=name, X=X, y=y)
    tstr = time_delta_now(t)
    logger.info(" -- %s -- Finished, took %s", name, tstr)


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="SPFlow Maxout Experiments")
    parser.add_argument("--result-dir", default="results", help="path to the result directory", metavar="DIR")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--synth-only", default=False, action="store_true")
    parser.add_argument("--only-plots", default=False, action="store_true")
    parser.add_argument("--min-instances-slice", default=25, type=int)
    parser.add_argument("--learning-rate", default=0.05, type=float)
    parser.add_argument("--experiment", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "iris-2d",
            "wine-2d",
            "diabetes",
            "audit",
            "banknotes",
            "ionosphere",
            "sonar",
            "wheat-2d",
            "synth-8-easy",
            "synth-64-easy",
            "synth-8-hard",
            "synth-64-hard",
        ],
    )
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
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    np.random.seed(1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Get args
    args = parse_args()

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
    if args.experiment == "loss":
        main_loss_experiments()
    elif args.experiment == "main":
        main_dataset_experiments()
    else:
        assert False, "Illegal experiment type: " + args.experiment
