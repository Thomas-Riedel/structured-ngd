# Reference: https://github.com/hollance/reliability-diagrams

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def _reliability_diagram_subplot(ax, bin_data,
                                 draw_ece=True,
                                 draw_mce=True,
                                 draw_bin_importance=False,
                                 title="Reliability Diagram",
                                 xlabel="Confidence",
                                 ylabel=None,
                                 plot_topk=False,
                                 top_k=5):
    if ylabel is None:
        ylabel = "Expected Accuracy"
        if plot_topk:
            ylabel = f"Expected Top--{top_k} Accuracy"
    if plot_topk:
        acc_label = f"Top--{top_k} Accuracy"
    else:
        acc_label = "Accuracy"
    """Draws a reliability diagram into a subplot."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["r_counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8 * normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1 * bin_size + 0.9 * bin_size*normalized_counts

    red = np.zeros((len(counts), 4))
    red[:, 0] = 240 / 255.
    red[:, 1] = 60 / 255.
    red[:, 2] = 60 / 255.
    red[:, 3] = alphas

    blue = np.zeros((len(counts), 4))
    blue[:, 0] = 76 / 255
    blue[:, 1] = 114 / 255
    blue[:, 2] = 176 / 255
    blue[:, 3] = alphas

    acc_plt = ax.bar(positions, accuracies, bottom=0, width=widths, linewidth=1,
                     edgecolor=blue, color=blue, label=acc_label)
    gap_plt = ax.bar(positions, np.abs(accuracies - confidences),
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor=red, color=red, linewidth=1, label="Gap")
    ax.bar(positions, 0, bottom=accuracies, width=widths,
           edgecolor="black", color="black", alpha=1.0, linewidth=3)
    ax.bar(positions, accuracies, bottom=0, width=widths, linewidth=1,
           edgecolor="black", color=(0, 0, 0, 0))

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    text = ''
    if plot_topk:
        if draw_ece:
            ece_topk = (bin_data["expected_calibration_error_topk"] * 100)
            text += f"Top--{top_k} ECE={ece_topk:5.2f}\n"
        if draw_mce:
            mce_topk = (bin_data["max_calibration_error_topk"] * 100)
            text += f"Top--{top_k} MCE={mce_topk:5.2f}"
    else:
        if draw_ece:
            ece = (bin_data["expected_calibration_error"] * 100)
            text += f"ECE={ece:5.2f}\n"
        if draw_mce:
            mce = (bin_data["max_calibration_error"] * 100)
            text += f"MCE={mce:5.2f}"
    if draw_ece or draw_mce:
        ax.text(0.98, 0.04, text, color="black", fontsize=14,
                ha="right", va="bottom", transform=ax.transAxes,
                bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    leg = ax.legend(handles=[acc_plt, gap_plt], loc='upper left', fontsize=16)
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)


def _confidence_histogram_subplot(ax, bin_data,
                                  draw_averages=True,
                                  title="Examples per bin",
                                  xlabel="Confidence",
                                  ylabel="\% of Samples"):
    """Draws a confidence histogram into a subplot."""
    counts = bin_data["r_counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    ax.bar(positions, counts / np.sum(np.abs(counts)), bottom=1, width=bin_size * 0.9)

    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_averages:
        acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=3,
                             c="black", label="Accuracy")
        conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=3,
                              c="#444", label="Avg. Confidence")
        ax.legend(handles=[acc_plt, conf_plt])


def _reliability_diagram_combined(bin_data, draw_ece, draw_mce, draw_bin_importance, draw_averages,
                                  title, plot_topk, figsize, dpi, return_fig):
    """Draws a reliability diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi,
                           gridspec_kw={"height_ratios": [4, 1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    _reliability_diagram_subplot(ax[0], bin_data, draw_ece, draw_mce, draw_bin_importance,
                                 title=title, plot_topk=plot_topk, xlabel="")

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["r_counts"]
    bin_data["r_counts"] = -bin_data["r_counts"]
    _confidence_histogram_subplot(ax[1], bin_data, draw_averages, title="")
    bin_data["r_counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    new_ticks = (100 - 100 * np.abs(ax[1].get_yticks()).round(2)).astype(int)
    ax[1].set_yticklabels(new_ticks)

    plt.tight_layout()

    if return_fig:
        return fig


def reliability_diagram(bin_data, draw_ece=True, draw_mce=True, draw_bin_importance='alpha',
                        draw_averages=True, title="Reliability Diagram",
                        figsize=(6, 6), dpi=100, return_fig=False, plot_topk=False):
    """Draws a reliability diagram and confidence histogram in a single plot.

    First, the model's predictions are divided up into bins based on their
    confidence scores.

    The reliability diagram shows the gap between average accuracy and average
    confidence in each bin. These are the red bars.

    The black line is the accuracy, the other end of the bar is the confidence.

    Ideally, there is no gap and the black line is on the dotted diagonal.
    In that case, the model is properly calibrated and we can interpret the
    confidence scores as probabilities.

    The confidence histogram visualizes how many examples are in each bin.
    This is useful for judging how much each bin contributes to the calibration
    error.

    The confidence histogram also shows the overall accuracy and confidence.
    The closer these two lines are together, the better the calibration.

    The ECE or Expected Calibration Error is a summary statistic that gives the
    difference in expectation between confidence and accuracy. In other words,
    it's a weighted average of the gaps across all bins. A lower ECE is better.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        draw_averages: whether to draw the overall accuracy and confidence in
            the confidence histogram
        title: optional title for the plot
        figsize: setting for matplotlib; height is ignored
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    # bin_data = compute_calibration(true_labels, pred_labels, confidences, num_bins)
    return _reliability_diagram_combined(bin_data, draw_ece, draw_mce, draw_bin_importance,
                                         draw_averages, title, plot_topk=plot_topk, figsize=figsize,
                                         dpi=dpi, return_fig=return_fig)


def _uncertainty_diagram_subplot(ax, bin_data,
                                 draw_uce=True,
                                 draw_muce=True,
                                 draw_bin_importance='alpha',
                                 title="Uncertainty Diagram",
                                 xlabel="Entropy",
                                 ylabel=None,
                                 plot_topk=False,
                                 top_k=5):
    if ylabel is None:
        ylabel = f"Expected Top--1 Error"
        if plot_topk:
            ylabel = f"Expected Top--{top_k} Error"
    if plot_topk:
        error_label = f"Top--{top_k} Error"
    else:
        error_label = "Top--1 Error"
    """Draws a reliability diagram into a subplot."""
    errors = bin_data["errors"]
    uncertainties = bin_data["uncertainties"]
    counts = bin_data["u_counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8 * normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1 * bin_size + 0.9 * bin_size*normalized_counts

    pink = np.zeros((len(counts), 4))
    pink[:, 0] = 245 / 255.
    pink[:, 1] = 97 / 255.
    pink[:, 2] = 221 / 255.
    pink[:, 3] = alphas

    green = np.zeros((len(counts), 4))
    green[:, 0] = 86 / 255
    green[:, 1] = 173 / 255
    green[:, 2] = 116 / 255
    green[:, 3] = alphas

    error_plt = ax.bar(positions, errors, bottom=0, width=widths, linewidth=1,
                       edgecolor=green, color=green, label=error_label)
    gap_plt = ax.bar(positions, np.abs(errors - uncertainties),
                     bottom=np.minimum(errors, uncertainties), width=widths,
                     edgecolor=pink, color=pink, linewidth=1, label="Gap")
    ax.bar(positions, 0, bottom=errors, width=widths,
           edgecolor="black", color="black", alpha=1.0, linewidth=3)
    ax.bar(positions, errors, bottom=0, width=widths, linewidth=1,
           edgecolor="black", facecolor=(0, 0, 0, 0))

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    text = ''
    if plot_topk:
        if draw_uce:
            uce_topk = (bin_data["expected_uncertainty_error_topk"] * 100)
            text += f"Top--{top_k} ECE={uce_topk:5.2f}\n"
        if draw_muce:
            muce_topk = (bin_data["max_uncertainty_error_topk"] * 100)
            text += f"Top--{top_k} MCE={muce_topk:5.2f}"
    else:
        if draw_uce:
            uce = (bin_data["expected_uncertainty_error"] * 100)
            text += f"UCE={uce:5.2f}\n"
        if draw_muce:
            muce = (bin_data["max_uncertainty_error"] * 100)
            text += f"MUCE={muce:5.2f}"
    if draw_uce or draw_muce:
        ax.text(0.98, 0.04, text, color="black", fontsize=14,
                ha="right", va="bottom", transform=ax.transAxes,
                bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    leg = ax.legend(handles=[error_plt, gap_plt], loc='upper left', fontsize=16)
    for lh in leg.legendHandles:
        lh.set_alpha(0.5)


def _uncertainty_histogram_subplot(ax, bin_data, draw_averages=True, title="Examples per bin",
                                   xlabel="Entropy", ylabel="\% of Samples"):
    """Draws an uncertainty histogram into a subplot."""
    counts = bin_data["u_counts"]
    bins = bin_data["bins"]

    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    green = (86/255, 173/255, 116/255)
    ax.bar(positions, counts / np.sum(np.abs(counts)), bottom=1, width=bin_size * 0.9, color=green)

    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_averages:
        error_plt = ax.axvline(x=bin_data["avg_error"], ls="solid", lw=3, c="black", label="Top--1 Error")
        uncert_plt = ax.axvline(x=bin_data["avg_uncertainty"], ls="dotted", lw=3, c="#444", label="Avg. Uncertainties")
        ax.legend(handles=[error_plt, uncert_plt])


def _uncertainty_diagram_combined(bin_data, draw_uce, draw_muce, draw_bin_importance, draw_averages,
                                  title, plot_topk, figsize, dpi, return_fig):
    """Draws an uncertainty diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi,
                           gridspec_kw={"height_ratios": [4, 1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    _uncertainty_diagram_subplot(ax[0], bin_data, draw_uce, draw_muce, draw_bin_importance, title=title, xlabel="")

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["u_counts"]
    bin_data["u_counts"] = -bin_data["u_counts"]
    _uncertainty_histogram_subplot(ax[1], bin_data, draw_averages, title="")
    bin_data["u_counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    new_ticks = (100 - 100 * np.abs(ax[1].get_yticks()).round(2)).astype(int)
    ax[1].set_yticklabels(new_ticks)

    plt.tight_layout()

    if return_fig:
        return fig


def uncertainty_diagram(bin_data, draw_uce=True, draw_muce=True, draw_bin_importance='alpha',
                        draw_averages=True, title="Uncertainty Diagram",
                        figsize=(6, 6), dpi=100, return_fig=False, plot_topk=False):
    """Draws a reliability diagram and confidence histogram in a single plot.

    First, the model's predictions are divided up into bins based on their
    confidence scores.

    The reliability diagram shows the gap between average accuracy and average
    confidence in each bin. These are the red bars.

    The black line is the accuracy, the other end of the bar is the confidence.

    Ideally, there is no gap and the black line is on the dotted diagonal.
    In that case, the model is properly calibrated and we can interpret the
    confidence scores as probabilities.

    The confidence histogram visualizes how many examples are in each bin.
    This is useful for judging how much each bin contributes to the calibration
    error.

    The confidence histogram also shows the overall accuracy and confidence.
    The closer these two lines are together, the better the calibration.

    The ECE or Expected Calibration Error is a summary statistic that gives the
    difference in expectation between confidence and accuracy. In other words,
    it's a weighted average of the gaps across all bins. A lower ECE is better.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        draw_averages: whether to draw the overall accuracy and confidence in
            the confidence histogram
        title: optional title for the plot
        figsize: setting for matplotlib; height is ignored
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    # bin_data = compute_calibration(true_labels, pred_labels, confidences, num_bins)
    return _uncertainty_diagram_combined(bin_data, draw_uce, draw_muce, draw_bin_importance,
                                         draw_averages, title, plot_topk=plot_topk, figsize=figsize,
                                         dpi=dpi, return_fig=return_fig)
