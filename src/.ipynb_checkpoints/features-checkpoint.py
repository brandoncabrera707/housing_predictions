import numpy as np
import pandas as pd
def add_basement_features(df):
  df = df.toCopy()
  df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
  df['sqft_basement_nonzero'] = df['sqft_basement'].where(df['sqft_basement']> 0, np.nan)
  return df