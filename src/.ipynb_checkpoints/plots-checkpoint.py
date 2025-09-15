import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
def plot_scatter_matrix(frame,attributes, alpha=0.5, figsize=None, ax=None, 
                        grid=False, diagonal='hist', marker='.', density_kwds=None, 
                        hist_kwds=None, range_padding=0.05, **kwargs):
  scatter_matrix(frame[attributes], figsize)
  plt.show()