"""
Plot results.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

from main import ensure_dir

# Apply seaborn default styling
sns.set()
dpi = 80
ftype = "csv"
dataset_names = ["iris-3d", "iris-2d", "wine-3d", "wine-2d", "audit"]
N_SYNTH_NFEAT_VALUES = [8, 16, 32, 64]
N_SYNTH_PINF_VALUES = [0.25, 1.0]
for nfeat in N_SYNTH_NFEAT_VALUES:
    for pinf in N_SYNTH_PINF_VALUES:
        dataset_names.append("synth-nfeat={}-pinf={}".format(nfeat, pinf))

WAF_FACTORS = [0.1, 0.3, 0.5]
suffixes = ["random-init", *["from-sum-weight-waf-%s" % waf for waf in WAF_FACTORS]]
suffixes_short = ["random-init", *["waf=%s" % waf for waf in WAF_FACTORS]]


def parse_args():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(description="SPFlow Maxout Experiments")

    parser.add_argument("--result-dir", default="results", help="path to the result directory", metavar="DIR")

    args = parser.parse_args()
    # Create result dir on the fly
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    return args


def plot_auc_exp():
    base_dir = os.path.join(args.result_dir, "auc-eval")
    plot_dir = os.path.join(plot_base_dir, "auc-eval")
    ensure_dir(plot_dir)

    for ds_name in dataset_names:
        fig = plt.figure()
        sub1 = fig.add_subplot(2, 2, 1)
        sub2 = fig.add_subplot(2, 2, 2)
        plt.title(ds_name)

        for suffix, suffix_short in zip(suffixes, suffixes_short):
            fname = "{}-{}.{}".format(ds_name, suffix, ftype)
            full_fname = os.path.join(base_dir, fname)
            x = np.loadtxt(full_fname, delimiter=",", comments="#")
            
            # Get data: epochs, train score, test score
            ks = x[:, 0]
            xs_train = x[:, 1]
            xs_test = x[:, 2]

            # Plot train on left subplot and test on right subplot
            sub1.scatter(ks, xs_train, label=suffix_short, marker="x", alpha=0.9)
            sub2.scatter(ks, xs_test, label=suffix_short, marker="x", alpha=0.9)

        # First subplot setup
        sub1.title(ds_name)
        sub1.xlabel("Number of internal nodes k")
        sub1.ylabel("AUC score")
        sub1.xscale("symlog", basex=2)
        sub1.legend(loc="lower right")
        sub2.title(ds_name)

        # Second subplot setup
        sub2.xlabel("Number of internal nodes k")
        sub2.ylabel("AUC score")
        sub2.xscale("symlog", basex=2)
        sub2.legend(loc="lower right")
        plt.savefig(os.path.join(plot_dir, "{}.pdf".format(ds_name)), dpi=dpi)


def plot_loss_exp():
    base_dir = os.path.join(args.result_dir, "loss-eval")
    plot_dir = os.path.join(plot_base_dir, "loss-eval")
    ensure_dir(plot_dir)

    for ds_name in dataset_names:
        for suffix in suffixes:
            plt.figure()
            title = "{} - {}: Loss Comparison".format(ds_name, suffix)

            fname = "{}-{}.{}".format(ds_name, suffix, ftype)
            full_fname = os.path.join(base_dir, fname)
            df = pd.read_csv(full_fname, sep=",", header=0)
            df = df.drop(columns="# epoch")
            df.columns = ["maxout=False"] + ["maxout=$%s$" % 2**i for i in range(1, 6)]
            ax = plt.gca()
            dfcut = df[np.abs(df-df.mean()) <= (3*df.std())]
            dfmin = dfcut.min().min()
            dfmax = dfcut.max().max()
            diff = dfmax - dfmin
            ax.set_ylim(dfmin - 0.05 * diff, dfmax + 0.05 * diff )
            df.plot(title=title, ax=ax)

            plt.xlabel("Epoch")
            plt.ylabel("Loss (log likelihood)")
            plt.savefig(os.path.join(plot_dir, "{}-{}.pdf".format(ds_name, suffix)), dpi=dpi)


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
            dfcut = df[np.abs(df-df.mean()) <= (3*df.std())]
            dfmin = dfcut.min().min()
            dfmax = dfcut.max().max()
            diff = dfmax - dfmin
            ax.set_ylim(dfmin - 0.05 * diff, dfmax + 0.05 * diff )
            df.plot(title=title, ax=ax)

            plt.xlabel("Epoch")
            plt.ylabel("Loss (log likelihood)")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "{}-{}.pdf".format(ds_name, suffix)), dpi=dpi)
            plt.close()

    pass

def generate_beamer_frames():
    dirs = [("loss-eval", "Loss Evaluation"), ("learning-rate-eval", "Learning Rate Evaluation")]
    with open("fnames.tex", "w") as f:
        lines = []

        # AUC scores
        tag = "AUC Evaluation"
        plot_dir = os.path.join(plot_base_dir, "auc-eval") 
        lines.append("\\begin{frame}{Experiment}")
        lines.append("\\LARGE")
        lines.append(tag)
        lines.append("\\end{frame}")
        for ds_name in dataset_names:
            fname = os.path.join(plot_dir, "{}".format(ds_name))
            lines.append("% {} %".format(ds_name))
            lines.append("\\begin{frame}{Results: %s\\\\%s}" % (ds_name, tag))
            lines.append("	\\vspace*{-2em}")
            lines.append("	\centering")
            lines.append("	\\begin{figure}")
            lines.append("		\includegraphics[width=0.7\\textwidth]{{../"+fname+"}.pdf}")	
            lines.append("	\end{figure}")
            lines.append("\end{frame}")

        # Loss/ LR eval
        for d, tag in dirs:
            plot_dir = os.path.join(plot_base_dir, d)

            lines.append("\\begin{frame}{Experiment}")
            lines.append("\\LARGE")
            lines.append(tag)
            lines.append("\\end{frame}")
            for ds_name in dataset_names:
                lines.append("\\begin{frame}{Results}")
                lines.append("\\LARGE")
                lines.append("Dataset: %s" % ds_name)
                lines.append("\\end{frame}")
                for suffix in suffixes:

                    fname = os.path.join(plot_dir, "{}-{}".format(ds_name, suffix))

                    lines.append("% {} {} %".format(ds_name, suffix))
                    lines.append("\\begin{frame}{Dataset: %s\\\\%s\\\\Setup: %s}" % (ds_name, tag, suffix))
                    lines.append("	\\vspace*{-2em}")
                    lines.append("	\centering")
                    lines.append("	\\begin{figure}")
                    lines.append("		\includegraphics[width=0.7\\textwidth]{{../"+fname+"}.pdf}")	
                    lines.append("	\end{figure}")
                    lines.append("\end{frame}")

        f.write("\n".join(lines))

if __name__ == "__main__":
    args = parse_args()
    plot_base_dir = os.path.join(args.result_dir, "plots")
    generate_beamer_frames()
    # Run plot generation
    plot_loss_exp()
    plot_lr_exp()
    plot_auc_exp()
