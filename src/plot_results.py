"""
Plot results.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from main import ensure_dir

ftype = "csv"
dataset_names = ["iris-3d", "iris-2d", "wine-3d", "wine-2d", "audit"]
dataset_names=[]
N_SYNTH_NFEAT_VALUES = [8, 16, 32, 64]
N_SYNTH_PINF_VALUES = [0.25, 1.0]
for nfeat in N_SYNTH_NFEAT_VALUES:
    for pinf in N_SYNTH_PINF_VALUES:
        dataset_names.append("synth-nfeat={}-pinf={}".format(nfeat, pinf))

WAF_FACTORS = [0.1, 0.3, 0.5]
suffixes = ["random-init", *["from-sum-weight-waf-%s" % waf for waf in WAF_FACTORS]]


def parse_args():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(description="SPFlow Maxout Experiments")

    parser.add_argument("--result-dir", default="results", help="path to the result directory", metavar="DIR")

    args = parser.parse_args()
    # Create result dir on the fly
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    return args


def plot_accuracy_exp():
    base_dir = os.path.join(args.result_dir, "accuracy-eval")
    plot_dir = os.path.join(plot_base_dir, "accuracy-eval")
    ensure_dir(plot_dir)

    for ds_name in dataset_names:
        plt.figure()
        plt.title(ds_name)

        for suffix in suffixes:
            fname = "{}-{}.{}".format(ds_name, suffix, ftype)
            full_fname = os.path.join(base_dir, fname)
            x = np.loadtxt(full_fname, delimiter=",", comments="#")
            ks = x[:, 0]
            xs = x[:, 1]
            plt.plot(ks, xs, label=suffix)

        plt.xlabel("Number of internal nodes k")
        plt.ylabel("Accuracy score")
        # plt.xscale("log", basex=2)
        plt.legend(loc="lower right")
        plt.show()


def plot_loss_exp():
    base_dir = os.path.join(args.result_dir, "loss-eval")
    plot_dir = os.path.join(plot_base_dir, "loss-eval")
    ensure_dir(plot_dir)

    for ds_name in dataset_names:
        for suffix in suffixes:
            plt.figure()
            plt.title("{} - {}: Loss Comparison".format(ds_name, suffix))

            fname = "{}-{}.{}".format(ds_name, suffix, ftype)
            full_fname = os.path.join(base_dir, fname)
            df = pd.read_csv(full_fname, sep=",", header=0)

            print(df.head())

            df.plot()

            plt.xlabel("Epoch")
            plt.ylabel("Loss (log likelihood)")
            plt.show()
            plt.savefig(os.path.join(plot_dir, "{}-{}.png".format(ds_name, suffix)))


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
            df.plot(title=title)

            plt.xlabel("Epoch")
            plt.ylabel("Loss (log likelihood)")
            plt.savefig(os.path.join(plot_dir, "{}-{}.png".format(ds_name, suffix)))
            plt.close()

    pass


if __name__ == "__main__":
    args = parse_args()
    plot_base_dir = os.path.join(args.result_dir, "plots")

    # Run plot generation
    # plot_accuracy_exp()
    # plot_loss_exp()
    plot_lr_exp()
