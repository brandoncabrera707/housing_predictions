import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def add_basement_features(df):
  """Adds a boolean feature representing whether a house has a basement(1) or doesn't have a basement(0)

  Args:
      df (DataFrame): DataFrame in which we will add boolean column to

  Returns:
      _DataFrame_: Original DataFrame with a new feature 'has_basement'
  """
  df['has_basement'] = (df['sqft_basement'] > 0).astype(int)
  return df


def encode_cities_val_test_set(df, encoder):
  """Uses one hot encoder to transform the validation and test sets

  Args:
      df (DataFrame): DataFrame that is going to be encoded
      encoder (OneHotEncoder): Fitted OneHotEncoder object

  Returns:
      DataFrame: the orignal DataFrame that was passed in, concatenated with the one hot encoded cities DataFrame 
  """

  one_hot_encoded_cities = encoder.transform(df[['city']]) #only transform the city column 
  one_hot_encoded_cities_df = pd.DataFrame(one_hot_encoded_cities, columns = encoder.get_feature_names_out(), index = df.index)
  df = pd.concat([df, one_hot_encoded_cities_df], axis = 1)
  df.drop('city', inplace = True, axis = 1) # No need for city column anymore
  return df

def apply_standard_scaler(df, scaler):
  """Applies standardization to the DataFrame passed into using the StandardScaler object passed into it

  Args:
      df (DataFrame): DataFrame that is going to be standardized
      scaler (StandardScaler): Fitted StandardScaler object we want to use on the dataset

  Returns:
      DataFrame: Returns a scaled version of the DataFrame 
  """

  df_nparray = scaler.transform(df) # only transform data since scaler is already fitted
  df_scaled = pd.DataFrame(df_nparray, index = df.index, columns = df.columns)
  return df_scaled

  
  
class logTransform(BaseEstimator, TransformerMixin):
  """Takes the logarithm of arguement X and returns it as a DataFrame using X's orignal index and columns

  Args:
      BaseEstimator (_type_): _description_
      TransformerMixin (_type_): _description_
  """
  def __init__(self):
    pass
  def fit(self, X, y = None):
    return self
  def transform(self, X):
    return pd.DataFrame(np.log(X), index = X.index, columns = X.columns)

class cbrtTransformer(BaseEstimator, TransformerMixin):
  def __init(self):
    """Empty constructor
    """
    pass
  def fit(self, X, y = None): 
    """Simply returns current object so that transform() can be used on it 
    """
    return self
  def transform(self, X):
    """Takes the cube root of argument X and returns it as a DataFrame using X's orignal index and columns

    Args:
        X (DataFrame): 

    Returns:
        DataFrame: returns the trans
    """
    return pd.DataFrame(np.cbrt(X), index = X.index, columns = X.columns)