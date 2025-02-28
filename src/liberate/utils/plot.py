from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import torch


def diff_distribution(
    diff: Union[np.ndarray, torch.Tensor], figsize=(12, 5), dpi=200
) -> plt.Figure:
    """
    Plot the distribution of errors in a histogram and box plot.

    Args:
        diff (Union[np.ndarray, torch.Tensor]): diff array containing the errors, such as diff = y_true - y_pred.
        figsize (tuple, optional): Figure size. Defaults to (12, 5).
        dpi (int, optional): Dots per inch for the figure. Defaults to 200.

    Returns:
        plt.Figure: Figure object containing the plots.
    """

    diff = np.array(diff).flatten()

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    axes[0].hist(diff, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0].set_title("Error Distribution (Histogram)", fontsize=16)
    axes[0].set_xlabel("Error", fontsize=14)
    axes[0].set_ylabel("Frequency", fontsize=14)

    axes[1].boxplot(
        diff,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightcoral"),
    )
    axes[1].set_title("Error Box Plot", fontsize=16)
    axes[1].set_ylabel("Error", fontsize=14)

    plt.tight_layout()
    return fig
