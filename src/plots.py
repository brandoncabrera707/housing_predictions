import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def plot_scatter_matrix_cols(
    frame,attributes,alpha=0.5,figsize=None,ax=None,grid=False,
    diagonal="hist",marker=".",density_kwds=None,hist_kwds=None,range_padding=0.05, **kwargs):
    """Wrapper around pandas scatter_matrix that adds attribute filtering"""

    axes = scatter_matrix(
        frame[attributes],figsize=figsize,alpha=alpha,ax=ax,grid=grid,diagonal=diagonal,
        marker=marker,density_kwds=density_kwds,hist_kwds=hist_kwds,range_padding=range_padding,
        **kwargs)
    plt.show()
    return axes