import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color='w')
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
    
def cm_heatmap(cm):
    fig, axs = plt.subplots(1, len(cm), figsize=(10,3))
    for i, k in enumerate(cm.keys()):
        heatmap(cm[k], [0,1], [0,1], ax=axs[i], cmap='jet')
        annotate_heatmap(axs[i].get_images()[0], data=cm[k], valfmt="{x:d}")
        axs[i].set_title(k)
        axs[i].set(xlabel='Model Prediction', ylabel='Ground Truth')

def calibration_plot(y, probs):
	fig = plt.figure(figsize=(10, 10))
	ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
	ax2 = plt.subplot2grid((3, 1), (2, 0))
	ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
	ax1.axhline(np.mean(y), c='r', linestyle=':', label="Baseline mean")

	for k in probs.keys():
	    frac_pos, mean_pred = calibration_curve(y, probs[k], n_bins=10)
	    ax1.plot(mean_pred, frac_pos, "s-", label=k)
	    ax2.hist(probs[k], range=(0, 1), bins=10, label=k, histtype='step', lw=2)
	
	ax1.set_xlabel("Predicted probability")
	ax1.set_ylabel("Fraction of positives")
	ax1.set_ylim([-0.05, 1.05])
	ax1.legend(loc="lower right")
	ax1.set_title('Calibration plots  (reliability curve)')
	ax2.set_xlabel("Mean predicted value")
	ax2.set_ylabel("Count")
	ax2.legend(loc="upper center", ncol=2)
	plt.tight_layout()
