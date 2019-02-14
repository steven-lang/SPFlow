"""
Plot results.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datasets
from typing import Dict

from main import ensure_dir

# Apply seaborn default styling
sns.set()


def get_num_sum_nodes() -> Dict[str, int]:
    """Get number of sum nodes in each spn architecture for a specific dataset"""
    res = {}
    for ds_name in dataset_names:
        # Get structure file name based on dataset name
        struct_fname = "plots/struct-plots/spn-{}-stats.txt".format(ds_name)
        try:
            with open(struct_fname, "r") as f:
                # Read all lines
                for line in f.readlines():
                    # Find line with "Sum"
                    if "sum" in line.lower():
                        # Extract number of sum nodes
                        n = line.split("=")[1].strip()
                        res[ds_name] = int(n)
        except Exception:
            print("Failed to load {}".format(struct_fname))

    return res


def parse_args():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(description="SPFlow Maxout Experiments")
    parser.add_argument("--result-dir", default="results", help="path to the result directory", metavar="DIR")
    parser.add_argument("--bic-score", action="store_true", help="Enable loss BIC score")
    args = parser.parse_args()
    # Create result dir on the fly
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    return args


def plot_auc_exp():
    base_dir = os.path.join(args.result_dir, "auc-epoch-eval")
    plot_dir = os.path.join(plot_base_dir, "auc-epoch-eval")
    ensure_dir(plot_dir)

    for ds_name in dataset_names:

        for suffix, suffix_short in zip(suffixes, suffixes_short):
            # Setup figure
            fig = plt.figure(figsize=(12, 9))
            sub1 = fig.add_subplot(2, 1, 1)
            sub2 = fig.add_subplot(2, 1, 2)
            plt.title(ds_name)

            # Get data
            fname_train = "{}-{}-train.{}".format(ds_name, suffix, ftype)
            fname_test = "{}-{}-test.{}".format(ds_name, suffix, ftype)
            full_fname_train = os.path.join(base_dir, fname_train)
            full_fname_test = os.path.join(base_dir, fname_test)
            x_train = np.loadtxt(full_fname_train, delimiter=",", comments="#")
            x_test = np.loadtxt(full_fname_test, delimiter=",", comments="#")

            x_train = x_train[0:100, :]
            x_test = x_test[0:100, :]

            epochs = x_train[:, 0]

            for i, col in enumerate(["False", 2, 3, 4, 5, 8, 16, 32]):
                # Plot train on left subplot and test on right subplot
                sub1.plot(epochs, x_train[:, i + 1], label="k=" + str(col), alpha=0.9)
                sub2.plot(epochs, x_test[:, i + 1], label="k=" + str(col), alpha=0.9)

            # First subplot setup
            sub1.set_title("Train Set")
            sub1.set_xlabel("Epoch")
            sub1.set_ylabel("AUC score")
            # sub1.xscale("symlog", basex=2)
            sub1.legend(loc="lower right")
            sub2.set_title("Test Set")

            # Second subplot setup
            sub2.set_xlabel("Number of internal nodes k")
            sub2.set_ylabel("AUC score")
            # sub2.set_xscale("symlog", basex=2)
            sub2.legend(loc="lower right")
            fig.suptitle("{}: {}".format(ds_name, suffix_short))
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(os.path.join(plot_dir, "{}-{}.png".format(ds_name, suffix_short)), dpi=dpi)


def plot_loss_exp():
    base_dir = os.path.join(args.result_dir, "loss-epoch-eval")
    plot_dir = os.path.join(plot_base_dir, "loss-epoch-eval")
    ensure_dir(plot_dir)

    for ds_name in dataset_names:
        for suffix in suffixes:
            plt.figure()
            title = "{} - {}: Loss Comparison".format(ds_name, suffix)

            fname = "{}-{}.{}".format(ds_name, suffix, ftype)
            full_fname = os.path.join(base_dir, fname)
            df = pd.read_csv(full_fname, sep=",", header=0)
            df = df.drop(columns="# epoch")
            df.columns = ["maxout=False"] + ["maxout=$%s$" % i for i in range(1, 6)]
            ax = plt.gca()
            dfcut = df[np.abs(df - df.mean()) <= (3 * df.std())]
            dfmin = dfcut.min().min()
            dfmax = dfcut.max().max()
            diff = dfmax - dfmin
            ax.set_ylim(dfmin - 0.05 * diff, dfmax + 0.05 * diff)
            df.plot(title=title, ax=ax)

            plt.xlabel("Epoch")
            plt.ylabel("Loss (mock log likelihood)")
            plt.savefig(os.path.join(plot_dir, "{}-{}.png".format(ds_name, suffix)), dpi=dpi)


def plot_epoch_loss_acc():
    base_dir_loss = os.path.join(args.result_dir, "loss-epoch-eval")
    base_dir_acc = os.path.join(args.result_dir, "acc-epoch-eval")
    plot_dir = os.path.join(plot_base_dir, "epoch-eval")
    ensure_dir(plot_dir)

    # For each dataset and each suffix (configuration)
    for ds_name in dataset_names:
        try:
            for suffix in suffixes:

                # Define file names
                acc_train_fname = "{}/{}-{}-train.{}".format(base_dir_acc, ds_name, suffix, ftype)
                acc_test_fname = "{}/{}-{}-test.{}".format(base_dir_acc, ds_name, suffix, ftype)
                loss_train_fname = "{}/{}-{}-train.{}".format(base_dir_loss, ds_name, suffix, ftype)
                loss_test_fname = "{}/{}-{}-test.{}".format(base_dir_loss, ds_name, suffix, ftype)

                # Load dataframes
                df_loss_train = pd.read_csv(loss_train_fname, sep=",", header=0)
                df_loss_test = pd.read_csv(loss_test_fname, sep=",", header=0)
                df_acc_train = pd.read_csv(acc_train_fname, sep=",", header=0)
                df_acc_test = pd.read_csv(acc_test_fname, sep=",", header=0)

                # Fix columns
                for df in [df_loss_train, df_loss_test, df_acc_train, df_acc_test]:
                    df.drop(columns="# epoch", inplace=True)
                    df.columns = ["maxout=False"] + ["maxout=$%s$" % i for i in [2, 3, 4, 5, 8, 16, 32, 64]]

                # Plot each df in its own subplot
                fig, axes = plt.subplots(nrows=2, ncols=2)
                fig.set_figheight(10)
                fig.set_figwidth(15)

                # Correct loss for BIC score: add d/2*log(N) where d: number of sum weights and N: number of samples
                n_sums = dataset_sum_nums[ds_name]
                n_data = dataset_size[ds_name]

                if args.bic_score:
                    Z_k1 = n_sums
                    for col, k in zip(df.columns, [1, 2, 3, 4, 5, 8, 16, 32, 64]):
                        Z_kk = n_sums * k
                        d = 1 - Z_k1 / Z_kk  # Relative BIC score
                        df_loss_train[col] += d / 2 * np.log(n_data * 2 / 3)  # 2/3 train
                        df_loss_test[col] += d / 2 * np.log(n_data * 1 / 3)  # 1/3 test

                for col in df_loss_train.columns:
                    axes[0, 0].plot(range(TF_N_EPOCHS), df_loss_train[col].values[:100])
                    axes[0, 1].plot(range(TF_N_EPOCHS), df_loss_test[col].values[:100])
                    axes[1, 0].plot(range(TF_N_EPOCHS), df_acc_train[col].values[:100])
                    axes[1, 1].plot(range(TF_N_EPOCHS), df_acc_test[col].values[:100])

                axes[0, 0].set_title("Train Loss")
                if args.bic_score:
                    axes[0, 0].set_ylabel("BIC Score")
                else:
                    axes[0, 0].set_ylabel("Log-Likelihood")
                axes[0, 1].set_title("Test Loss")
                # axes[0, 1].set_ylabel("Log-Likelihood")
                axes[1, 0].set_title("Train Accuracy")
                axes[1, 0].set_ylabel("Accuracy")
                axes[1, 0].set_xlabel("Epochs")
                axes[1, 1].set_title("Test Accuracy")
                # axes[1, 1].set_ylabel("Accuracy")
                axes[1, 1].set_xlabel("Epochs")
                title = "{} - {}:  Loss/Accuracy over Epochs".format(ds_name, suffix)
                fig.suptitle(title)
                plt.legend(labels=[c for c in df.columns])
                plt.savefig(os.path.join(plot_dir, "{}-{}.png".format(ds_name, suffix)), dpi=dpi)
        except Exception as e:
            print("Error in Dataset:", ds_name)
            print(e)
            print(repr(e))


def plot_epoch_loss_auc():
    base_dir_loss = os.path.join(args.result_dir, "loss-epoch-eval")
    base_dir_auc = os.path.join(args.result_dir, "auc-epoch-eval")
    plot_dir = os.path.join(plot_base_dir, "epoch-eval")
    ensure_dir(plot_dir)

    # For each dataset and each suffix (configuration)
    for ds_name in dataset_names:
        for suffix in suffixes:

            # Define file names
            auc_train_fname = "{}/{}-{}-train.{}".format(base_dir_auc, ds_name, suffix, ftype)
            auc_test_fname = "{}/{}-{}-test.{}".format(base_dir_auc, ds_name, suffix, ftype)
            loss_fname = "{}/{}-{}.{}".format(base_dir_loss, ds_name, suffix, ftype)

            # Load dataframes
            df_loss = pd.read_csv(loss_fname, sep=",", header=0)
            df_auc_train = pd.read_csv(auc_train_fname, sep=",", header=0)
            df_auc_test = pd.read_csv(auc_test_fname, sep=",", header=0)

            # Fix columns
            for df in [df_loss, df_auc_train, df_auc_test]:
                df.drop(columns="# epoch", inplace=True)
                df.columns = ["maxout=False"] + ["maxout=$%s$" % i for i in [2, 3, 4, 5, 8, 16, 32]]

            # Plot each df in its own subplot
            fig, axes = plt.subplots(nrows=3, ncols=1)
            fig.set_figheight(10)
            fig.set_figwidth(10)

            for c in df_loss.columns:
                axes[0].plot(range(TF_N_EPOCHS), df_loss[c].values[:100])
                axes[1].plot(range(TF_N_EPOCHS), df_auc_train[c].values[:100])
                axes[2].plot(range(TF_N_EPOCHS), df_auc_test[c].values[:100])

            axes[0].set_title("Loss")
            axes[0].set_ylabel("Log-Likelihood")
            axes[1].set_title("Train Set")
            axes[1].set_ylabel("AUC")
            axes[2].set_title("Test Set")
            axes[2].set_ylabel("AUC")
            axes[2].set_xlabel("Epochs")
            title = "{} - {}: Epoch Comparison (Loss, AUC)".format(ds_name, suffix)
            fig.suptitle(title)
            plt.legend(labels=[c for c in df.columns])
            plt.savefig(os.path.join(plot_dir, "{}-{}.png".format(ds_name, suffix)), dpi=dpi)


def plot_lr_exp():
    base_dir = os.path.join(args.result_dir, "learning-rate-eval")
    plot_dir = os.path.join(plot_base_dir, "learning-rate-eval")
    ensure_dir(plot_dir)

    for ds_name in dataset_names:
        for suffix in suffixes:
            plt.figure()

            title = "{} - {}: Learning Rate Comparison".format(ds_name, suffix)
            fname = "{}-{}.{}".format(ds_name, suffix, ftype)
            full_fname = os.path.join(base_dir, fname)
            df = pd.read_csv(full_fname, sep=",", header=0)
            df = df.drop(columns="# epoch")
            df.columns = ["maxout=False"] + ["lr=$2^{-%s}$" % i for i in range(1, 10)]
            ax = plt.gca()
            dfcut = df[np.abs(df - df.mean()) <= (3 * df.std())]
            dfmin = dfcut.min().min()
            dfmax = dfcut.max().max()
            diff = dfmax - dfmin
            ax.set_ylim(dfmin - 0.05 * diff, dfmax + 0.05 * diff)
            df.plot(title=title, ax=ax)

            plt.xlabel("Epoch")
            plt.ylabel("Loss (mock log likelihood)")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "{}-{}.png".format(ds_name, suffix)), dpi=dpi)
            plt.close()


if __name__ == "__main__":
    dpi = 120
    ftype = "csv"
    dataset_names = [
        "iris-2d",
        "wine-2d",
        "diabetes",
        "audit",
        "banknotes",
        "ionosphere",
        "sonar",
        "wheat-2d",
        "synth-8-easy",
        "synth-8-hard",
        "synth-64-easy",
        "synth-64-hard",
    ]
    dataset_size = {}
    for name, loader in datasets.load_dataset_map().items():
        X, y = loader()
        n = X.shape[0]
        dataset_size[name] = n

    dataset_sum_nums = get_num_sum_nodes()

    # TODO: Remove next line
    # dataset_names = []
    # N_SYNTH_NFEAT_VALUES = [8, 64]
    # N_SYNTH_PINF_VALUES = [0.25, 1.0]
    TF_N_EPOCHS = 100
    # for nfeat in N_SYNTH_NFEAT_VALUES:
    #     for pinf in N_SYNTH_PINF_VALUES:
    #         dataset_names.append("synth-nfeat={}-pinf={}".format(nfeat, pinf))

    WAF_FACTORS = [0.1]
    suffixes = ["random-init", "augment-sum-init"]
    suffixes_short = suffixes

    args = parse_args()
    plot_base_dir = os.path.join("plots", args.result_dir)
    # Run plot generation
    # plot_epoch_loss_auc()
    plot_epoch_loss_acc()
    # plot_lr_exp()
    # plot_auc_exp()
