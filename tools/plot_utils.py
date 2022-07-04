# Modified from https://github.com/facebookresearch/detr
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'),
              ewm_col=0, log_name='metrics.json'):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if not dir.exists():
            raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            df.rolling(60).mean()[::60].interpolate().ewm(com=ewm_col).mean().plot(
                y=field,
                ax=axs[j],
                color=color,
                style='-')
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)

    return fig, axs


def save_plot(log_path_list, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), save_name='result.png'):
    if isinstance(log_path_list, list):
        log = [Path(log_path) for log_path in log_path_list]
    else:
        log = Path(log_path_list)
    fig, _ = plot_logs(log, fields)
    fig.savefig(os.path.join(log_path_list[0], save_name))


if __name__ == '__main__':
    save_plot(['OSFormer',
               'OSFormer-ZEROLIKE',
               'OSFormer-NNEMBEDDING'],
              fields=('total_loss', 'loss_ins'))
